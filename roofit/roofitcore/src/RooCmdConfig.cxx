/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/


/**
\file RooCmdConfig.cxx
\class RooCmdConfig
\ingroup Roofitcore

Configurable parser for RooCmdArg named
arguments. It maps the contents of named arguments named to integers,
doubles, strings and TObjects that can be retrieved after processing
a set of RooCmdArgs. The parser also has options to enforce syntax
rules such as (conditionally) required arguments, mutually exclusive
arguments and dependencies between arguments.
**/

#include <RooCmdConfig.h>
#include <RooMsgService.h>

#include <ROOT/StringUtils.hxx>

#include <iostream>


namespace {

template<class Collection>
typename Collection::const_iterator findVar(Collection const& coll, const char * name) {
  return std::find_if(coll.begin(), coll.end(), [name](auto const& v){ return v.name == name; });
}

}


using namespace std;

ClassImp(RooCmdConfig);


////////////////////////////////////////////////////////////////////////////////
/// Constructor taking descriptive name of owner/user which
/// is used as prefix for any warning or error messages
/// generated by this parser

RooCmdConfig::RooCmdConfig(RooStringView methodName) : _name(methodName)
{
  _rList.SetOwner() ;
  _fList.SetOwner() ;
  _mList.SetOwner() ;
  _yList.SetOwner() ;
  _pList.SetOwner() ;
}


namespace {

void cloneList(TList const& inList, TList & outList) {
  outList.SetOwner(true);
  for(auto * elem : inList) {
    outList.Add(elem->Clone()) ;
  }
}

} // namespace


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooCmdConfig::RooCmdConfig(const RooCmdConfig &other)
   : TObject(other),
     _name(other._name),
     _verbose(other._verbose),
     _error(other._error),
     _allowUndefined(other._allowUndefined),
     _iList(other._iList),
     _dList(other._dList),
     _sList(other._sList),
     _oList(other._oList),
     _cList(other._cList)
{
  cloneList(other._rList, _rList); // Required cmd list
  cloneList(other._fList, _fList); // Forbidden cmd list
  cloneList(other._mList, _mList); // Mutex cmd list
  cloneList(other._yList, _yList); // Dependency cmd list
  cloneList(other._pList, _pList); // Processed cmd list
}



////////////////////////////////////////////////////////////////////////////////
/// Return string with names of arguments that were required, but not
/// processed

std::string RooCmdConfig::missingArgs() const
{
  std::string ret;

  bool first = true;
  for(TObject * s : _rList) {
    if (first) {
      first=false ;
    } else {
      ret += ", ";
    }
    ret += static_cast<TObjString*>(s)->String();
  }

  return ret;
}



////////////////////////////////////////////////////////////////////////////////
/// Define that processing argument name refArgName requires processing
/// of argument named neededArgName to successfully complete parsing

void RooCmdConfig::defineDependency(const char* refArgName, const char* neededArgName)
{
  _yList.Add(new TNamed(refArgName,neededArgName)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Define integer property name 'name' mapped to integer in slot 'intNum' in RooCmdArg with name argName
/// Define default value for this int property to be defVal in case named argument is not processed

bool RooCmdConfig::defineInt(const char* name, const char* argName, int intNum, int defVal)
{
  if (findVar(_iList, name) != _iList.end()) {
    coutE(InputArguments) << "RooCmdConfig::defineInt: name '" << name << "' already defined" << endl ;
    return true ;
  }

  _iList.emplace_back();
  auto& ri = _iList.back();
  ri.name = name;
  ri.argName = argName;
  ri.val = defVal;
  ri.num = intNum;
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Define double property name 'name' mapped to double in slot 'doubleNum' in RooCmdArg with name argName
/// Define default value for this double property to be defVal in case named argument is not processed

bool RooCmdConfig::defineDouble(const char* name, const char* argName, int doubleNum, double defVal)
{
  if (findVar(_dList, name) != _dList.end()) {
    coutE(InputArguments) << "RooCmdConfig::defineDouble: name '" << name << "' already defined" << endl ;
    return true ;
  }

  _dList.emplace_back();
  auto& rd = _dList.back();
  rd.name = name;
  rd.argName = argName;
  rd.val = defVal;
  rd.num = doubleNum;
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Define double property name 'name' mapped to double in slot 'stringNum' in RooCmdArg with name argName
/// Define default value for this double property to be defVal in case named argument is not processed
/// If appendMode is true, values found in multiple matching RooCmdArg arguments will be concatenated
/// in the output string. If it is false, only the value of the last processed instance is retained

bool RooCmdConfig::defineString(const char* name, const char* argName, int stringNum, const char* defVal, bool appendMode)
{
  if (findVar(_sList, name) != _sList.end()) {
    coutE(InputArguments) << "RooCmdConfig::defineString: name '" << name << "' already defined" << endl ;
    return true ;
  }

  _sList.emplace_back();
  auto& rs = _sList.back();
  rs.name = name;
  rs.argName = argName;
  rs.val = defVal;
  rs.appendMode = appendMode;
  rs.num = stringNum;
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Define TObject property name 'name' mapped to object in slot 'setNum' in RooCmdArg with name argName
/// Define default value for this TObject property to be defVal in case named argument is not processed.
/// If isArray is true, an array of TObjects is harvested in case multiple matching named arguments are processed.
/// If isArray is false, only the TObject in the last processed named argument is retained

bool RooCmdConfig::defineObject(const char* name, const char* argName, int setNum, const TObject* defVal, bool isArray)
{

  if (findVar(_oList, name) != _oList.end()) {
    coutE(InputArguments) << "RooCmdConfig::defineObject: name '" << name << "' already defined" << endl ;
    return true ;
  }

  _oList.emplace_back();
  auto& os = _oList.back();
  os.name = name;
  os.argName = argName;
  os.val.Add(const_cast<TObject*>(defVal));
  os.appendMode = isArray;
  os.num = setNum;
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Define TObject property name 'name' mapped to object in slot 'setNum' in RooCmdArg with name argName
/// Define default value for this TObject property to be defVal in case named argument is not processed.
/// If isArray is true, an array of TObjects is harvested in case multiple matching named arguments are processed.
/// If isArray is false, only the TObject in the last processed named argument is retained

bool RooCmdConfig::defineSet(const char* name, const char* argName, int setNum, const RooArgSet* defVal)
{

  if (findVar(_cList, name) != _cList.end()) {
    coutE(InputArguments) << "RooCmdConfig::defineObject: name '" << name << "' already defined" << endl ;
    return true ;
  }

  _cList.emplace_back();
  auto& cs = _cList.back();
  cs.name = name;
  cs.argName = argName;
  cs.val = const_cast<RooArgSet*>(defVal);
  cs.num = setNum;
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print configuration of parser

void RooCmdConfig::print() const
{
  // Find registered integer fields for this opcode
  for(auto const& ri : _iList) {
    cout << ri.name << "[int] = " << ri.val << endl ;
  }

  // Find registered double fields for this opcode
  for(auto const& rd : _dList) {
    cout << rd.name << "[double] = " << rd.val << endl ;
  }

  // Find registered string fields for this opcode
  for(auto const& rs : _sList) {
    cout << rs.name << "[string] = \"" << rs.val << "\"" << endl ;
  }

  // Find registered argset fields for this opcode
  for(auto const& ro : _oList) {
    cout << ro.name << "[TObject] = " ;
    auto const * obj = ro.val.At(0);
    if (obj) {
      cout << obj->GetName() << endl ;
    } else {

      cout << "(null)" << endl ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Process given list with RooCmdArgs

bool RooCmdConfig::process(const RooLinkedList& argList)
{
  bool ret(false) ;
  for(auto * arg : static_range_cast<RooCmdArg*>(argList)) {
    ret |= process(*arg) ;
  }
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Process given RooCmdArg

bool RooCmdConfig::process(const RooCmdArg& arg)
{
  // Retrieve command code
  const char* opc = arg.opcode() ;

  // Ignore empty commands
  if (!opc) return false ;

  // Check if not forbidden
  if (_fList.FindObject(opc)) {
    coutE(InputArguments) << _name << " ERROR: argument " << opc << " not allowed in this context" << endl ;
    _error = true ;
    return true ;
  }

  // Check if this code generates any dependencies
  TObject* dep = _yList.FindObject(opc) ;
  if (dep) {
    // Dependent command found, add to required list if not already processed
    if (!_pList.FindObject(dep->GetTitle())) {
      _rList.Add(new TObjString(dep->GetTitle())) ;
      if (_verbose) {
   cout << "RooCmdConfig::process: " << opc << " has unprocessed dependent " << dep->GetTitle()
        << ", adding to required list" << endl ;
      }
    } else {
      if (_verbose) {
   cout << "RooCmdConfig::process: " << opc << " dependent " << dep->GetTitle() << " is already processed" << endl ;
      }
    }
  }

  // Check for mutexes
  TObject * mutex = _mList.FindObject(opc) ;
  if (mutex) {
    if (_verbose) {
      cout << "RooCmdConfig::process: " << opc << " excludes " << mutex->GetTitle()
      << ", adding to forbidden list" << endl ;
    }
    _fList.Add(new TObjString(mutex->GetTitle())) ;
  }


  bool anyField(false) ;

  // Find registered integer fields for this opcode
  for(auto& ri : _iList) {
    if (!TString(opc).CompareTo(ri.argName)) {
      ri.val = arg.getInt(ri.num) ;
      anyField = true ;
      if (_verbose) {
   cout << "RooCmdConfig::process " << ri.name << "[int]" << " set to " << ri.val << endl ;
      }
    }
  }

  // Find registered double fields for this opcode
  for(auto& rd : _dList) {
    if (!TString(opc).CompareTo(rd.argName)) {
      rd.val = arg.getDouble(rd.num) ;
      anyField = true ;
      if (_verbose) {
   cout << "RooCmdConfig::process " << rd.name << "[double]" << " set to " << rd.val << endl ;
      }
    }
  }

  // Find registered string fields for this opcode
  for(auto& rs : _sList) {
    if (rs.argName == opc) {

      // RooCmdArg::getString can return nullptr, so we have to protect against this
      auto const * newStr = arg.getString(rs.num);

      if (!rs.val.empty() && rs.appendMode) {
        rs.val += ",";
        rs.val += newStr ? newStr : "(null)";
      } else {
        if(newStr) rs.val = newStr;
      }
      anyField = true ;
      if (_verbose) {
        std::cout << "RooCmdConfig::process " << rs.name << "[string]" << " set to " << rs.val << std::endl ;
      }
    }
  }

  // Find registered TObject fields for this opcode
  for(auto& os : _oList) {
    if (!TString(opc).CompareTo(os.argName)) {
      if(!os.appendMode) os.val.Clear();
      os.val.Add(const_cast<TObject*>(arg.getObject(os.num)));
      anyField = true ;
      if (_verbose) {
   cout << "RooCmdConfig::process " << os.name << "[TObject]" << " set to " ;
   if (os.val.At(0)) {
     cout << os.val.At(0)->GetName() << endl ;
   } else {
     cout << "(null)" << endl ;
   }
      }
    }
  }

  // Find registered RooArgSet fields for this opcode
  for(auto& cs : _cList) {
    if (!TString(opc).CompareTo(cs.argName)) {
      cs.val = const_cast<RooArgSet*>(arg.getSet(cs.num));
      anyField = true ;
      if (_verbose) {
   cout << "RooCmdConfig::process " << cs.name << "[RooArgSet]" << " set to " ;
   if (cs.val) {
     cout << cs.val->GetName() << endl ;
   } else {
     cout << "(null)" << endl ;
   }
      }
    }
  }

  bool multiArg = !TString("MultiArg").CompareTo(opc) ;

  if (!anyField && !_allowUndefined && !multiArg) {
    coutE(InputArguments) << _name << " ERROR: unrecognized command: " << opc << endl ;
  }


  // Remove command from required-args list (if it was there)
  TObject* obj;
  while ( (obj = _rList.FindObject(opc)) ) {
    _rList.Remove(obj);
  }

  // Add command the processed list
  TNamed *pcmd = new TNamed(opc,opc) ;
  _pList.Add(pcmd) ;

  bool depRet = false ;
  if (arg.procSubArgs()) {
    for (int ia=0 ; ia<arg.subArgs().GetSize() ; ia++) {
      RooCmdArg* subArg = static_cast<RooCmdArg*>(arg.subArgs().At(ia)) ;
      if (strlen(subArg->GetName())>0) {
   RooCmdArg subArgCopy(*subArg) ;
   if (arg.prefixSubArgs()) {
     subArgCopy.SetName(Form("%s::%s",arg.GetName(),subArg->GetName())) ;
   }
   depRet |= process(subArgCopy) ;
      }
    }
  }

  return ((anyField||_allowUndefined)?false:true)||depRet ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return true if RooCmdArg with name 'cmdName' has been processed

bool RooCmdConfig::hasProcessed(const char* cmdName) const
{
  return _pList.FindObject(cmdName) ? true : false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return integer property registered with name 'name'. If no
/// property is registered, return defVal

int RooCmdConfig::getInt(const char* name, int defVal) const
{
  auto found = findVar(_iList, name);
  return found != _iList.end() ? found->val : defVal;
}



////////////////////////////////////////////////////////////////////////////////
/// Return double property registered with name 'name'. If no
/// property is registered, return defVal

double RooCmdConfig::getDouble(const char* name, double defVal) const
{
  auto found = findVar(_dList, name);
  return found != _dList.end() ? found->val : defVal;
}



////////////////////////////////////////////////////////////////////////////////
/// Return string property registered with name 'name'. If no
/// property is registered, return defVal. If convEmptyToNull
/// is true, empty string will be returned as null pointers

const char* RooCmdConfig::getString(const char* name, const char* defVal, bool convEmptyToNull) const
{
  auto found = findVar(_sList, name);
  if(found == _sList.end()) return defVal;
  return (convEmptyToNull && found->val.empty()) ? nullptr : found->val.c_str();
}



////////////////////////////////////////////////////////////////////////////////
/// Return TObject property registered with name 'name'. If no
/// property is registered, return defVal

TObject* RooCmdConfig::getObject(const char* name, TObject* defVal) const
{
  auto found = findVar(_oList, name);
  return found != _oList.end() ? found->val.At(0) : defVal ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return RooArgSet property registered with name 'name'. If no
/// property is registered, return defVal

RooArgSet* RooCmdConfig::getSet(const char* name, RooArgSet* defVal) const
{
  auto found = findVar(_cList, name);
  return found != _cList.end() ? found->val : defVal ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return list of objects registered with name 'name'

const RooLinkedList& RooCmdConfig::getObjectList(const char* name) const
{
  const static RooLinkedList defaultDummy ;
  auto found = findVar(_oList, name);
  return found != _oList.end() ? found->val : defaultDummy ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return true of parsing was successful

bool RooCmdConfig::ok(bool verbose) const
{
  if (_rList.GetSize()==0 && !_error) return true ;

  if (verbose) {
    std::string margs = missingArgs() ;
    if (!margs.empty()) {
      coutE(InputArguments) << _name << " ERROR: missing arguments: " << margs << endl ;
    } else {
      coutE(InputArguments) << _name << " ERROR: illegal combination of arguments and/or missing arguments" << endl ;
    }
  }
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function that strips command names listed (comma separated) in cmdsToPurge from cmdList

void RooCmdConfig::stripCmdList(RooLinkedList& cmdList, const char* cmdsToPurge)
{
  // Sanity check
  if (!cmdsToPurge) return ;

  // Copy command list for parsing
  for(auto const& name : ROOT::Split(cmdsToPurge, ",")) {
    if (TObject* cmd = cmdList.FindObject(name.c_str())) {
      cmdList.Remove(cmd);
    }
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Utility function to filter commands listed in cmdNameList from cmdInList. Filtered arguments are put in the returned list.
/// If removeFromInList is true then these commands are removed from the input list

RooLinkedList RooCmdConfig::filterCmdList(RooLinkedList& cmdInList, const char* cmdNameList, bool removeFromInList) const
{
  RooLinkedList filterList ;
  if (!cmdNameList) return filterList ;

  // Copy command list for parsing
  for(auto const& name : ROOT::Split(cmdNameList, ",")) {
    if (TObject* cmd = cmdInList.FindObject(name.c_str())) {
      if (removeFromInList) {
        cmdInList.Remove(cmd) ;
      }
      filterList.Add(cmd) ;
    }
  }
  return filterList ;
}



////////////////////////////////////////////////////////////////////////////////
/// Find a given double in a list of RooCmdArg.
/// Should only be used to initialise base classes in constructors.
double RooCmdConfig::decodeDoubleOnTheFly(const char* callerID, const char* cmdArgName, int idx, double defVal,
    std::initializer_list<std::reference_wrapper<const RooCmdArg>> args) {
  RooCmdConfig pc(callerID);
  pc.allowUndefined();
  pc.defineDouble("theDouble", cmdArgName, idx, defVal);
  pc.process(args.begin(), args.end());
  return pc.getDouble("theDouble");
}
