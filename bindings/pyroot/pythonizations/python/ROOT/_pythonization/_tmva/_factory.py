import ROOT
from ROOT import TMVA

from ._utils import _kwargs_to_Tmvacmdargs, cpp_signature

class factory(object):
    r"""Some member functions of factory that take a TmvaCmdArg as argument also support keyword arguments.
    This applies to factory::BookMethod().

    """
    @cpp_signature(
        "Factory::BookMethod( DataLoader *loader, TString theMethodName, TString methodTitle, TString theOption = "" );"
    )
    def BookMethod(self, *args, **kwargs):
        r"""factory::BookMethod() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        args, kwargs = _kwargs_to_Tmvacmdargs(*args, **kwargs)
        kwargs.split(":")
        return self._BookMethod(*args, **kwargs)
