#!/usr/bin/env python3
import json


def generate_json(opt, mol):
    js = {}

    js["keywords"] = opt.keywords

    # need to strip non-builtins e.g. numpy arrays
    js["initial_molecule"] = json.loads(mol.json())

    js["input_specification"] = opt.qc_spec.dict()
    js["input_specification"]["keywords"] = {
        "properties": [
            "dipole",
            "quadrupole",
            "wiberg_lowdin_indices",
            "mayer_indices",
        ],
    }

    js["input_specification"]["model"] = {
        "basis": opt.qc_spec.basis,
        "method": opt.qc_spec.method,
    }
    js["input_specification"].pop("basis")
    js["input_specification"].pop("method")

    js["memory"] = "2GB"
    js["nthreads"] = 1

    return js


def qca_query(oid, mid):
    import qcfractal.interface as ptl

    client = ptl.FractalClient()

    opt = client.query_procedures(oid)[0]

    if mid is None:
        mid = opt.initial_molecule
    mid = client.query_molecules(mid)[0]

    return opt, mid


def qca_configure_runtime(js, memory="2GB", nthreads=1, maxiter=200):
    if memory is not None:
        js["memory"] = memory
    if nthreads is not None:
        js["nthreads"] = int(nthreads)
    if maxiter is not None:
        js["keywords"]["maxiter"] = int(maxiter)
    return js


def qca_generate_input(oid, mid):
    opt, mol = qca_query(oid, mid)
    js = generate_json(opt, mol)
    return js 

def qca_run_geometric_opt_native(in_json_dict, file_prefix):
    """ Take a input dictionary loaded from json, and return an output dictionary for json """

    import logging
    import geometric
    import pkg_resources
    import logging.config
    from geometric.run_json import parse_input_json_dict, make_constraints_string

    input_opts = parse_input_json_dict(in_json_dict)

    logIni = pkg_resources.resource_filename(geometric.optimize.__name__, "log.ini")

    logger = logging.getLogger()
    if file_prefix is not None:
        logging.config.fileConfig(
            logIni, defaults={"logfilename": file_prefix + ".log"}, disable_existing_loggers=False
        )
        input_opts['xyzout'] = file_prefix + ".xyz"
    else:
        FORMAT = "%(message)s"
        logging.basicConfig(format=FORMAT)
        # All logging lines in geometric add their own newlines
        logger.handlers[0].terminator = ""
        logging.warning("Output file not given. Optimization trajectory will not be saved\n")

    M, engine = geometric.optimize.get_molecule_engine(**input_opts)

    # Get initial coordinates in bohr
    coords = M.xyzs[0].flatten() * geometric.nifty.ang2bohr

    # Read in the constraints
    constraints_dict = input_opts.get("constraints", {})
    if "scan" in constraints_dict:
        raise KeyError(
            "The constraint 'scan' keyword is not yet supported by the JSON interface"
        )

    constraints_string = make_constraints_string(constraints_dict)
    Cons, CVals = None, None
    if constraints_string:
        Cons, CVals = geometric.optimize.ParseConstraints(M, constraints_string)

    # set up the internal coordinate system
    coordsys = input_opts.get("coordsys", "tric")
    CoordSysDict = {
        "cart": (geometric.internal.CartesianCoordinates, False, False),
        "prim": (geometric.internal.PrimitiveInternalCoordinates, True, False),
        "dlc": (geometric.internal.DelocalizedInternalCoordinates, True, False),
        "hdlc": (geometric.internal.DelocalizedInternalCoordinates, False, True),
        "tric": (geometric.internal.DelocalizedInternalCoordinates, False, False),
    }

    CoordClass, connect, addcart = CoordSysDict[coordsys.lower()]
    IC = CoordClass(
        M,
        build=True,
        connect=connect,
        addcart=addcart,
        constraints=Cons,
        cvals=CVals[0] if CVals is not None else None,
    )

    # Print out information about the coordinate system
    if isinstance(IC, geometric.internal.CartesianCoordinates):
        logger.info("%i Cartesian coordinates being used\n" % (3 * M.na))
    else:
        logger.info(
            "%i internal coordinates being used (instead of %i Cartesians)\n"
            % (len(IC.Internals), 3 * M.na)
        )
    logger.info(IC)
    logger.info("\n")

    params = geometric.optimize.OptParams(**input_opts)

    # Run the optimization
    if Cons is None:
        # Run a standard geometry optimization
        geometric.optimize.Optimize(coords, M, IC, engine, None, params)
    else:
        # Run a constrained geometry optimization
        if isinstance(
            IC,
            (
                geometric.internal.CartesianCoordinates,
                geometric.internal.PrimitiveInternalCoordinates,
            ),
        ):
            raise RuntimeError(
                "Constraints only work with delocalized internal coordinates"
            )
        for ic, CVal in enumerate(CVals):
            if len(CVals) > 1:
                logger.info(
                    "---=== Scan %i/%i : Constrained Optimization ===---\n"
                    % (ic + 1, len(CVals))
                )
            IC = CoordClass(
                M,
                build=True,
                connect=connect,
                addcart=addcart,
                constraints=Cons,
                cvals=CVal,
            )
            IC.printConstraints(coords, thre=-1)
            geometric.optimize.Optimize(coords, M, IC, engine, None, params)


def qca_run_geometric_opt(js):
    import geometric

    out = geometric.run_json.geometric_run_json(js)
    return out


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "optimization",
        help="QCA ID of the optimization to run or JSON filename if -r specified",
    )
    parser.add_argument(
        "molecule_id",
        nargs="?",
        default=None,
        help="QCA ID of the molecule to use; default will use molecule for given optimization",
    )
    parser.add_argument("-o", "--out_file", default=None, help="Output file name")
    parser.add_argument(
        "-i",
        "--inputs-only",
        action="store_true",
        help="just generate input json; do not run (edit and then run with -r)",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=str,
        default=None,
        help="amount of memory to use, eg '10GB'",
    )
    parser.add_argument(
        "-n",
        "--nthreads",
        type=int,
        default=None, help="number of processors to use",
    )
    parser.add_argument(
        "-r",
        "--run-json",
        action="store_true",
        help="The optimization input is a JSON filename",
    )
    parser.add_argument(
        "-l",
        "--maxiter",
        type=int,
        default=None,
        help="maximum optimization iterations to perform",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Use the json interface rather than the native interface (to stdout)",
    )

    args = parser.parse_args()

    if args.run_json:
        js = json.load(open(args.optimization))
    else:
        js = qca_generate_input(args.optimization, args.molecule_id)

    js = qca_configure_runtime(js, args.memory, args.nthreads, args.maxiter)

    ret = ""
    if args.inputs_only:
        ret = js
    else:
        if args.json:
            ret = qca_run_geometric_opt(js)
        else:
            qca_run_geometric_opt_native(js, args.out_file)
    if len(ret) > 0:
        if args.out_file is None:
            print(json.dumps(ret, indent=2))
        elif (args.json or args.inputs_only):
            with open(args.out_file, "w") as fid:
                json.dump(ret, fid, indent=2)


if __name__ == "__main__":
    main()
