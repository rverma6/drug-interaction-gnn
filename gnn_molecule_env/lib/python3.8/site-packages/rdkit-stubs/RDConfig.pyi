"""
 Configuration for the RDKit Python code

"""
from __future__ import annotations
import os as os
import rdkit as rdkit
import sqlite3 as sqlite3
import sys as sys
__all__ = ['ObsoleteCodeError', 'RDCodeDir', 'RDContribDir', 'RDDataDatabase', 'RDDataDir', 'RDDocsDir', 'RDProjDir', 'RDTestDatabase', 'UnimplementedCodeError', 'defaultDBPassword', 'defaultDBUser', 'molViewer', 'os', 'pythonExe', 'pythonTestCommand', 'rdkit', 'rpcTestPort', 'sqlite3', 'sys', 'usePgSQL', 'useSqlLite']
class ObsoleteCodeError(Exception):
    pass
class UnimplementedCodeError(Exception):
    pass
RDCodeDir: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-10.9-x86_64-cpython-38/rdkit_install/lib/python3.8/site-packages/rdkit'
RDContribDir: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-10.9-x86_64-cpython-38/rdkit_install/share/RDKit/Contrib'
RDDataDatabase: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-10.9-x86_64-cpython-38/rdkit_install/share/RDKit/Data/RDData.sqlt'
RDDataDir: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-10.9-x86_64-cpython-38/rdkit_install/share/RDKit/Data'
RDDocsDir: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-10.9-x86_64-cpython-38/rdkit_install/share/RDKit/Docs'
RDProjDir: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-10.9-x86_64-cpython-38/rdkit_install/share/RDKit/Projects'
RDTestDatabase: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-10.9-x86_64-cpython-38/rdkit_install/share/RDKit/Data/RDTests.sqlt'
defaultDBPassword: str = 'masterkey'
defaultDBUser: str = 'sysdba'
molViewer: str = 'PYMOL'
pythonExe: str = '/private/var/folders/6f/9bq719ks5ws4blpr063kpj840000gn/T/cibw-run-_vew_9yk/cp38-macosx_x86_64/build/venv/bin/python'
pythonTestCommand: str = 'python'
rpcTestPort: int = 8423
usePgSQL: bool = False
useSqlLite: bool = True
