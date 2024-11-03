"""
 Descriptors derived from a molecule's 3D structure

"""
from __future__ import annotations
from rdkit.Chem.Descriptors import _isCallable
from rdkit.Chem import rdMolDescriptors
__all__ = ['CalcMolDescriptors3D', 'descList', 'rdMolDescriptors']
def CalcMolDescriptors3D(mol, confId = None):
    """
    
        Compute all 3D descriptors of a molecule
        
        Arguments:
        - mol: the molecule to work with
        - confId: conformer ID to work with. If not specified the default (-1) is used
        
        Return:
        
        dict
            A dictionary with decriptor names as keys and the descriptor values as values
    
        raises a ValueError 
            If the molecule does not have conformers
        
    """
def _setupDescriptors(namespace):
    ...
descList: list  # value = [('PMI1', <function <lambda> at 0x7fe87b7aa550>), ('PMI2', <function <lambda> at 0x7fe87c7cca60>), ('PMI3', <function <lambda> at 0x7fe87c7ccaf0>), ('NPR1', <function <lambda> at 0x7fe87c7ccb80>), ('NPR2', <function <lambda> at 0x7fe87c7ccc10>), ('RadiusOfGyration', <function <lambda> at 0x7fe87c7ccca0>), ('InertialShapeFactor', <function <lambda> at 0x7fe87c7ccd30>), ('Eccentricity', <function <lambda> at 0x7fe87c7ccdc0>), ('Asphericity', <function <lambda> at 0x7fe87c7cce50>), ('SpherocityIndex', <function <lambda> at 0x7fe87c7ccee0>), ('PBF', <function <lambda> at 0x7fe87c7ccf70>)]
