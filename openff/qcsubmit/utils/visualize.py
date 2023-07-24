import copy
from typing import List, Optional

from openff.toolkit.topology import Molecule
from openff.toolkit.utils.toolkits import OPENEYE_AVAILABLE, RDKIT_AVAILABLE
from typing_extensions import Literal


def molecules_to_pdf(
    molecules: List[Molecule],
    file_name: str,
    columns: int = 4,
    toolkit: Optional[Literal["openeye", "rdkit"]] = None,
):
    """Create a pdf file of the molecules with any driven dihedrals (as specified by
    including the 'dihedrals' in the molecules property dictionary) highlighted using
    either openeye or rdkit.

    Parameters:
        molecules: The molecules to include in the PDF.
        file_name: The name of the pdf file which will be produced.
        columns: The number of molecules per row.
        toolkit: The backend toolkit to use when producing the pdf file.
    """

    toolkits = {
        "openeye": (OPENEYE_AVAILABLE, _create_openeye_pdf),
        "rdkit": (RDKIT_AVAILABLE, _create_rdkit_pdf),
    }

    if toolkit is not None:
        if toolkit.lower() not in toolkits:
            raise ValueError(
                f"The requested toolkit backend: {toolkit} is not supported, chose "
                f"from {toolkits.keys()}"
            )

        toolkits = {toolkit: toolkits[toolkit]}

    for toolkit in toolkits:
        available, pdf_func = toolkits[toolkit]
        if available:
            return pdf_func(molecules, file_name, columns)

    raise ImportError(
        "No backend toolkit was found to generate the pdf please install openeye "
        "and/or rdkit."
    )


def _create_openeye_pdf(molecules: List[Molecule], file_name: str, columns: int):
    """Make the pdf of the molecules using OpenEye."""

    from openeye import oechem, oedepict

    itf = oechem.OEInterface()
    suppress_h = True
    rows = 10
    cols = columns
    ropts = oedepict.OEReportOptions(rows, cols)
    ropts.SetHeaderHeight(25)
    ropts.SetFooterHeight(25)
    ropts.SetCellGap(2)
    ropts.SetPageMargins(10)
    report = oedepict.OEReport(ropts)
    cellwidth, cellheight = report.GetCellWidth(), report.GetCellHeight()
    opts = oedepict.OE2DMolDisplayOptions(
        cellwidth, cellheight, oedepict.OEScale_Default * 0.5
    )
    opts.SetAromaticStyle(oedepict.OEAromaticStyle_Circle)
    pen = oedepict.OEPen(oechem.OEBlack, oechem.OEBlack, oedepict.OEFill_On, 1.0)
    opts.SetDefaultBondPen(pen)
    oedepict.OESetup2DMolDisplayOptions(opts, itf)

    # now we load the molecules
    for off_mol in molecules:
        off_mol = copy.deepcopy(off_mol)
        off_mol._conformers = []
        off_mol.name = None

        cell = report.NewCell()
        mol = off_mol.to_openeye()
        oedepict.OEPrepareDepiction(mol, False, suppress_h)
        disp = oedepict.OE2DMolDisplay(mol, opts)

        if "dihedrals" in off_mol.properties:
            # work out if we have a double or single torsion
            if len(off_mol.properties["dihedrals"]) == 1:
                dihedrals = off_mol.properties["dihedrals"][0]
                center_bonds = dihedrals[1:3]
            else:
                # double torsion case
                dihedrals = [
                    *off_mol.properties["dihedrals"][0],
                    *off_mol.properties["dihedrals"][1],
                ]
                center_bonds = [
                    *off_mol.properties["dihedrals"][0][1:3],
                    *off_mol.properties["dihedrals"][1][1:3],
                ]

            # Highlight element of interest
            class NoAtom(oechem.OEUnaryAtomPred):
                def __call__(self, atom):
                    return False

            class AtomInTorsion(oechem.OEUnaryAtomPred):
                def __call__(self, atom):
                    return atom.GetIdx() in dihedrals

            class NoBond(oechem.OEUnaryBondPred):
                def __call__(self, bond):
                    return False

            class CentralBondInTorsion(oechem.OEUnaryBondPred):
                def __call__(self, bond):
                    return (bond.GetBgn().GetIdx() in center_bonds) and (
                        bond.GetEnd().GetIdx() in center_bonds
                    )

            atoms = mol.GetAtoms(AtomInTorsion())
            bonds = mol.GetBonds(NoBond())
            abset = oechem.OEAtomBondSet(atoms, bonds)
            oedepict.OEAddHighlighting(
                disp,
                oechem.OEColor(oechem.OEYellow),
                oedepict.OEHighlightStyle_BallAndStick,
                abset,
            )

            atoms = mol.GetAtoms(NoAtom())
            bonds = mol.GetBonds(CentralBondInTorsion())
            abset = oechem.OEAtomBondSet(atoms, bonds)
            oedepict.OEAddHighlighting(
                disp,
                oechem.OEColor(oechem.OEOrange),
                oedepict.OEHighlightStyle_BallAndStick,
                abset,
            )

        oedepict.OERenderMolecule(cell, disp)

    oedepict.OEWriteReport(file_name, report)


def _create_rdkit_pdf(molecules: List[Molecule], file_name: str, columns: int):
    """
    Make the pdf of the molecules using rdkit.
    """
    from rdkit.Chem import AllChem, Draw

    rdkit_molecules = []
    tagged_atoms = []
    images = []
    for off_mol in molecules:
        rdkit_mol = off_mol.to_rdkit()
        AllChem.Compute2DCoords(rdkit_mol)
        rdkit_molecules.append(rdkit_mol)

        if "dihedrals" in off_mol.properties:
            tagged_atoms.extend(off_mol.properties["dihedrals"])
        else:
            tagged_atoms.append(tuple())

    # if no atoms are to be tagged set to None
    if not tagged_atoms:
        tagged_atoms = None

    # evey 24 molecules split the page
    for i in range(0, len(rdkit_molecules), 24):
        mol_chunk = rdkit_molecules[i : i + 24]
        if tagged_atoms is not None:
            tag_chunk = tagged_atoms[i : i + 24]
        else:
            tag_chunk = None

        # now make the image
        image = Draw.MolsToGridImage(
            mol_chunk,
            molsPerRow=columns,
            subImgSize=(500, 500),
            highlightAtomLists=tag_chunk,
        )
        # write the pdf to bytes and pass straight to the pdf merger
        images.append(image)

    images[0].save(file_name, append_images=images[1:], save_all=True)
