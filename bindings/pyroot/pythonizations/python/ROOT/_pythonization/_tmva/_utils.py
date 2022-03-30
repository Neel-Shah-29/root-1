
def _kwargs_to_Tmvacmdargs(*args, **kwargs):
    """Helper function to check kwargs with keys that correspond to a function that creates TmvaCmdArg."""

    def getter(k, v):
        # helper function to get CmdArg attribute from `TMVA`
        # Parameters:
        # k: key of the kwarg
        # v: value of the kwarg

        # We have to use ROOT here and not cppy.gbl, because the TMVA namespace is pythonized itself.
        import ROOT
        from ROOT import TMVA
        import libcppyy
        func = getattr(ROOT.TMVA, k)

        if isinstance(func, libcppyy.CPPOverload):
            # Pythonization for functions that don't pass any TmvaCmdArgs . For Eg,

            if "()" in func.func_doc:
                if not isinstance(v, bool):
                    raise TypeError("The keyword argument " + k + " can only take bool values.")
                return func() if v else ROOT.TmvaCmdArg.none()

        try:
            # If the keyword argument value is a tuple, list, set, or dict, first
            # try to unpack it as parameters to the TmvaCmdArg-generating
            # function. If this doesn't succeed, the tuple, list, or dict,
            # will be passed directly to the function as it's only argument.
            if isinstance(v, (tuple, list, set)):
                return func(*v)
            elif isinstance(v, (dict,)):
                return func(**v)
        except:
            pass

        return func(v)

    if kwargs:
        args = args + tuple((getter(k, v) for k, v in kwargs.items()))
    return args, {}

def cpp_signature(sig):
    """Decorator to set the `_cpp_signature` attribute of a function.
    This information can be used to generate the documentation.
    """

    def decorator(func):
        func._cpp_signature = sig
        return func

    return decorator
