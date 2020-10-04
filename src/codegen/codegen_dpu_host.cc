/*!
 * \file codegen_dpu_host.cc
 */
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "codegen_dpu_host.h"
#include "build_common.h"
#include "iostream"
#include <fstream> 
#include <tvm/make_api.h>
#include "../pass/ir_util.h"
#include <stdio.h>
#include <unistd.h>


//std::vector<tvm::Var> tmp = args_list;
namespace tvm {
//int len = (tvm::args_list).size();
//std::cout<< "arg_list size is : "<< len << std::endl;
namespace codegen {


CodeGenDPUHost::CodeGenDPUHost() {
  module_name_ = GetUniqueName("__tvm_module_ctx");
}

void CodeGenDPUHost::Init(bool output_ssa, bool emit_asserts) {
  emit_asserts_ = emit_asserts;
  CodeGenC::Init(output_ssa);
}

std::string dpuKernelNamePre (std::string kernelName) {
	std::string newKernelName = kernelName;
	int len = int(kernelName.length());
	if (len > 8) {
		std::string strPre = kernelName.substr(0,8);
		if (strPre == "fused_nn") {
			strPre = "DPU";
			newKernelName = strPre + kernelName.substr(8, (len - 8));
		}
	}
	return newKernelName;
}

bool dpuContain(std::string src, std::string str) {
	bool ret = false;
	int len1 = int(src.length()), len2 = int(str.length());
	int i = 0, j = 0;
	while (i < len1 && j < len2) {
		if (src[i] == str[j]) {
			ret = true;
			i++;
			j++;
		}
		else {
			ret = false;
			i++;
			j = 0;
		}
		
		continue;
	}
	if (j == len2 && ret) return true;
	else return false;
}


void CodeGenDPUHost::AddFunction(LoweredFunc f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();
  // add to alloc buffer type.
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.dtype());
  }

  functionName = dpuKernelNamePre(f->name);
  //functionName = f->name;

	char *dpuBuffer;
  if((dpuBuffer = getcwd(NULL, 0)) == NULL){
	  perror("getcwd error");
  }
  dpuCodeLibPath = dpuBuffer;
	if (dpuContain(functionName, "reshape")) {
		dpuCodeLibPath += R"(/dpuCodeUnuseLib/)" + functionName + R"(.c)";
	}
	dpuCodeLibPath += R"(/dpuCodeLib/)" + functionName + R"(.c)";
	free(dpuBuffer);
	
	dpufile.open(dpuCodeLibPath, std::ios::out);
	
  hasPrintedArgsList = false;
  args_list = f->args_list;
	argsListMap.clear();
	argsVec.clear();
	
  for (size_t i = 0; i < args_list.size(); ++i) {
    Var v = args_list[i];
    std::string argsName_ = v->name_hint;
    if (!argsListMap.count(argsName_)){
      argsListMap[argsName_] = true;
    }
  }

	for (size_t i = 0; i < f->args_list.size(); ++i) {
			Var v = f->args_list[i];
			std::string vid = AllocVarID(v.get());
			argsVec.emplace_back(vid);
	}

	
	for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    std::string vid = AllocVarID(v.get());
  }
	
  //this->stream << "\n\n\n";
  //this->stream << "void callfunction (";
  //stream << ") {\n";

	if (!hasPrintedArgsList) {
      //PrintIndent();
      //this->stream << functionName << "_kernel(";
      //for (size_t i = 0; i < argsVec.size(); ++i) {
        //Var v = argsVec[i];
      //  std::string vid = argsVec[i];
        //vid = GetUniqueName(vid);
      //  if (i != 0) stream << ", ";
      //    stream << vid;
    // }
     //this->stream <<  ");\n";
     //this->stream <<"}"<<"\n\n\n";
     /* breif/ generate kernel_function's args list
      *   e.g. "void DPUGemm_kernel(float* A, float* B, float* C) {"
      */
     this->stream <<"---------------------DEVICE_CODE------------------------\n\n\n";
     this->stream << "void " << functionName << "_kernel(";
	   dpufile << "void " << functionName << "_kernel(";
     for (size_t i = 0; i < args_list.size(); ++i) {
        Var v = args_list[i];
        //can not use "AllocVarID(v.get())" here
        std::string vid = argsVec[i];
        //vid = GetUniqueName(vid);
        if (i != 0) { stream << ", "; dpufile << ", ";}
        if (v.dtype().is_handle()) {
          auto it = alloc_storage_scope_.find(v.get());
          if (it != alloc_storage_scope_.end()) {
            PrintStorageScope(it->second, stream);
        }
        if (i != 0) { stream << ' '; dpufile << ' ';}
        if (handle_data_type_.count(v.get())) {
            PrintType(handle_data_type_.at(v.get()), stream);
        } else {
            stream << "void";
						dpufile << "void";
        }
       stream << "*";
			 dpufile << "*";
       } else {
          PrintType(v.dtype(), stream);
       }
        stream << ' ' << vid;
			  dpufile << ' ' << vid;
      }
      stream << ") {\n";
		  dpufile << ") {\n";
      hasPrintedArgsList = true;
   }
	
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  //this->PrintIndent();
  this->stream << "}\n\n";
	dpufile << "}\n\n";
	dpufile.close();
}

std::string CodeGenDPUHost::Finish() {
  return CodeGenC::Finish();
}

void CodeGenDPUHost::PrintVecStore(const VarNode* buffer,
			                              DataType t, PrimExpr base,
			                              const std::string& value) {
  std::string ref = GetBufferRef(t, buffer, base);
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  stream << ref << " = " << value << ";\n";
	dpufile << ref << " = " << value << ";\n";
}

 void CodeGenDPUHost::PrintVecElemLoad(const std::string& vec,
		 																		 DataType t, int i,
		 																		 std::ostream& os) {	// NOLINT(*)
	 os << vec << ".s" << std::hex << i << std::dec;
	 dpufile << vec << ".s" << std::hex << i << std::dec;
 }


void CodeGenDPUHost::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1)
        << "does not support vector types";
    os << "void*";
		dpufile << "void*";
		return;
  }
  if (t == DataType::Bool()) {
    os << "bool"; 
		dpufile << "bool";
		return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16: {
	        os << "half";
			    dpufile << "half";
	        break;
      	}
      case 32: {
					os << "float"; 
					dpufile << "float";
					break;
      	}
      case 64:{
	        os << "double";
					dpufile << "double";
	        break;
      	}
      default: {fail = true; break;}
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
			dpufile << lanes;
			return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
			dpufile << 'u';
    }
    switch (t.bits()) {
      case 8: {os << "int8_t"; dpufile << "int8_t"; break;}
      case 16: {os << "int16_t"; dpufile << "int16_t"; break;}
      /***wjq***/
      case 32: {dpufile << "int";
								//dpufile.flush(); 
								os << "int";
								//os.flush();
			break;}
     /***wjq***/
      case 64: {
				os << "int64_t"; 
			  dpufile << "int64_t"; 
			  break;
			}
      case 1: {os << "int32_t"; dpufile << "int32_t"; break;}
      default: {fail = true; break;}
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes; dpufile << lanes; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

void CodeGenDPUHost::VisitExpr_(const BroadcastNode* op, std::ostream& os) {   // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "((";
	dpufile << "((";
  PrintType(op->dtype, os);
  os << ")(";
	dpufile << ")(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) { os << ", "; dpufile << ", "; }
    os << v;
		dpufile << v;
  }
  os << "))";
	dpufile << "))";
}

void CodeGenDPUHost::PrintGetFuncFromBackend(const std::string& func_name,
                                           const std::string& packed_func_name) {
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  this->stream << "if (" << packed_func_name << " == NULL) {\n";
	dpufile << "if (" << packed_func_name << " == NULL) {\n";
  int packed_func_if_scope = this->BeginScope();
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  this->stream << "if (TVMBackendGetFuncFromEnv(" << module_name_
              << ", \"" << func_name << "\""
              << ", &" << packed_func_name << ") != 0) {\n";
	dpufile << "if (TVMBackendGetFuncFromEnv(" << module_name_
              << ", \"" << func_name << "\""
              << ", &" << packed_func_name << ") != 0) {\n";
  int get_func_env_scope = this->BeginScope();
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  this->stream << "return -1;\n";
	dpufile << "return -1;\n";
  this->EndScope(get_func_env_scope);
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  this->stream << "}\n";
	dpufile << "}\n";
  this->EndScope(packed_func_if_scope);
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  this->stream << "}\n";
	dpufile << "}\n";
	//dpufile.close();
}

void CodeGenDPUHost::PrintFuncCall(const std::string& packed_func_name, int num_args) {
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  std::string ret_val = GetUniqueName("ret_val");
  std::string ret_type_code = GetUniqueName("ret_type_code");
  this->stream << "TVMValue " << ret_val << ";\n";
	dpufile << "TVMValue " << ret_val << ";\n";
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  this->stream << "int " << ret_type_code << ";\n";
	dpufile << "int " << ret_type_code << ";\n";
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  this->stream << "if (TVMFuncCall(" << packed_func_name << ", "
               << "(TVMValue*) stack_value" << ", " << "(int*) stack_tcode" << ", "
               << num_args << ", " << "&" << ret_val << ", " << "&"
               << ret_type_code << ") != 0) {\n";
	dpufile << "if (TVMFuncCall(" << packed_func_name << ", "
               << "(TVMValue*) stack_value" << ", " << "(int*) stack_tcode" << ", "
               << num_args << ", " << "&" << ret_val << ", " << "&"
               << ret_type_code << ") != 0) {\n";
  int func_call_scope = this->BeginScope();
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  this->stream << "return -1;\n";
	dpufile << "return -1;\n";
  this->EndScope(func_call_scope);
  this->PrintIndent();
	this->PrintDPUIndent(dpufile);
  this->stream << "}\n";
	dpufile << "}\n";
}

void CodeGenDPUHost::VisitExpr_(const CallNode *op, std::ostream& os) { // NOLINT(*)
  if (op->is_intrinsic(intrinsic::tvm_stack_alloca)) {
    std::string stack_name = GetUniqueName("stack");
    const std::string& type = op->args[0].as<StringImmNode>()->value;
    const IntImmNode* num = op->args[1].as<IntImmNode>();
    CHECK(num != nullptr);
    static_assert(alignof(TVMValue) % alignof(TVMArray) == 0, "invariant");
    size_t unit = sizeof(TVMValue);
    size_t size = 0;
    if (type == "shape") {
      size = (num->value * sizeof(tvm_index_t) + unit - 1) / unit;
    } else if (type == "arg_value") {
      size = (num->value * sizeof(TVMValue) + unit - 1) / unit;
    } else if (type == "arg_tcode") {
      size = (num->value * sizeof(int) + unit - 1) / unit;
    } else if (type == "array") {
      size = (num->value * sizeof(TVMArray) + unit - 1) / unit;
    } else {
      LOG(FATAL) << "Unknown stack alloca type " << type;
    }
    this->PrintIndent();
		this->PrintDPUIndent(dpufile);
    this->stream << "TVMValue " << stack_name << "[" << size << "];\n";
		dpufile << "TVMValue " << stack_name << "[" << size << "];\n";
    os << stack_name;
		dpufile << stack_name;
  } else if (op->is_intrinsic(intrinsic::tvm_call_packed_lowered)) {
    const StringImmNode* s = op->args[0].as<StringImmNode>();
    CHECK(s != nullptr) << "tvm_call_packed_lowered expects first argument as function name";
    int64_t begin = op->args[3].as<IntImmNode>()->value;
    int64_t end = op->args[4].as<IntImmNode>()->value;
    int64_t num_args = end - begin;
    CHECK_GE(num_args, 0);
    std::string func_name = s->value;
    std::string packed_func_name = GetUniqueName(func_name + "_packed");
    decl_stream << "static void* " << packed_func_name << " = NULL;\n";
		dpufile << "static void* " << packed_func_name << " = NULL;\n";
    this->PrintGetFuncFromBackend(func_name, packed_func_name);
    this->PrintFuncCall(packed_func_name, num_args);
  } else if (op->is_intrinsic(intrinsic::tvm_throw_last_error)) {
    this->PrintIndent();
		this->PrintDPUIndent(dpufile);
    this->stream << "return -1;\n";
		dpufile << "return -1;\n";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

/**********wjq**********/
void CodeGenDPUHost::VisitStmt_(const LetStmtNode* op) {
  //std::string value = PrintExpr(op->value);
	std::string value = "";
  if (print_ssa_form_) {
    CHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } 
  else {
    //PrintIndent();
    if (op->var.dtype() == DataType::Handle() &&
        handle_data_type_.count(op->var.get())) {
     // PrintType(handle_data_type_.at(op->var.get()), stream);
     // auto tmp =op->var.get();
      auto tmp =op->var.get();
		  if (!argsListMap.count(tmp->name_hint)) {
      std::string str = AllocVarID(op->var.get());
		  }
     //if (argsListMap.count(tmp->name_hint)) {
     //  argsVec.push_back(str);
       //PrintIndent();
       //PrintType(handle_data_type_.at(op->var.get()), stream);
       //stream << "* "
       //       << str
       //       << " = (";
       //PrintType(handle_data_type_.at(op->var.get()), stream);
       //stream << "*)"  << value << ";\n";
    // }
      //stream << "* "
      //       << AllocVarID(op->var.get())
      //       << " = (";
     // PrintType(handle_data_type_.at(op->var.get()), stream);
      //stream << "*)"  << value << ";\n";
    } else {
     // PrintType(op->var.dtype(), this->stream);
      auto tmp =op->var.get();
		  if (!argsListMap.count(tmp->name_hint)) {
      std::string str = AllocVarID(op->var.get());
		  }
      //if (argsListMap.count(tmp->name_hint)) {
       // argsVec.push_back(str);
        //PrintIndent();
        //PrintType(handle_data_type_.at(op->var.get()), stream);
        //stream << "* "
        //       << str
        //       << " = (";
        //PrintType(handle_data_type_.at(op->var.get()), stream);
        //stream << "*)"  << value << ";\n";
     //}
      //this->stream << ' '
      //             << AllocVarID(op->var.get())
      //             << " = " << value << ";\n";
    }
  }

  PrintStmt(op->body);
}
/*
void CodeGenDPUHost::VisitExpr_(const LoadNode* op, std::ostream& os) {  // NOLINT(*)
  int lanes = op->dtype.lanes();
  // delcare type.
  if (op->dtype.lanes() == 1) {
    std::string ref = GetBufferRef(op->dtype, op->buffer_var.get(), op->index);
    os << ref;
		dpufile << ref;
  } else {
    CHECK(is_one(op->predicate))
        << "predicated load is not supported";
    PrimExpr base;
    if (GetRamp1Base(op->index, op->dtype.lanes(), &base)) {
      std::string ref = GetVecLoad(op->dtype, op->buffer_var.get(), base);
      os << ref;
			dpufile << ref;
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // load seperately.
      std::string svalue = GetUniqueName("_");
      this->PrintIndent();
			this->PrintDPUIndent(dpufile);
      this->PrintType(op->dtype, stream);
      stream << ' ' << svalue << ";\n";
			dpufile << ' ' << svalue << ";\n";
      std::string sindex = SSAGetID(PrintExpr(op->index), op->index.dtype());
      std::string vid = GetVarID(op->buffer_var.get());
      DataType elem_type = op->dtype.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (op->buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
              value_temp << ' ';
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, op->index.dtype(), i, value_temp);
        value_temp << ']';
        PrintVecElemStore(svalue, op->dtype, i, value_temp.str());
      }
      os << svalue;
      EndScope(vec_scope);
    }
  }
}
*/

void CodeGenDPUHost::VisitStmt_(const AllocateNode* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  /***wjq***/
  auto tmp = op->buffer_var;
  if (!argsListMap.count(tmp->name_hint)) {
    args_list.push_back(tmp);
  } 
  argsVec.push_back(vid);
  
  if (op->new_expr.defined()) {
    // Prefer global static allocation for the program
    CHECK_EQ(op->free_function, "nop");
    std::string new_data = PrintExpr(op->new_expr);
    //this->PrintIndent();
    PrintType(op->dtype, stream);
    stream << "* "<< vid << '=' << new_data << ";\n";
		dpufile << "* "<< vid << '=' << new_data << ";\n";
  } else {
    this->PrintIndent();
		this->PrintDPUIndent(dpufile);
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";
    const VarNode* buffer = op->buffer_var.as<VarNode>();
    std::string scope = alloc_storage_scope_.at(buffer);
    PrintStorageScope(scope, stream);
    stream << ' ';
		dpufile << ' ';
    PrintType(op->dtype, stream);
    stream << ' '<< vid << '['
           << constant_size << "];\n";
		dpufile << ' '<< vid << '['
           << constant_size << "];\n";
  }
  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}


void CodeGenDPUHost::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == ir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      if (!var_idmap_.count(iv->var.get())) {
        BindThreadIndex(iv);
      }
    }
  } else if (op->attr_key == ir::attr::storage_scope) {
    const VarNode* v = op->node.as<VarNode>();
    CHECK(v);
    alloc_storage_scope_[v] = op->value.as<StringImmNode>()->value;
  } else if (op->attr_key == ir::attr::volatile_scope) {
    const VarNode* v = op->node.as<VarNode>();
    CHECK(v);
    //volatile_buf_.insert(v);
  }
	
	else if ((op->attr_key).substr(0,8) == ir::attr::dpu_pragma) {
		PrintIndent();
		PrintDPUIndent(dpufile);
		stream << op->attr_key << "\n";
	  dpufile << op->attr_key << "\n";
	}
	
  this->PrintStmt(op->body);
}


void CodeGenDPUHost::VisitStmt_(const ForNode* op) {
  
 
	/* breif/ generate call_function's args list, only one kernel function can be handled!
	 * param: hasPrintedArgsList: bool, used to determine if the kernel function needs to be printed
	 * param: args_list: Array<Var>, store the kernel args list
	 * param: args_names: string, function name
	 *   e.g. "DPUGemm_kernel(A, M, K, B, N, C);"
	 */

  
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
	PrintDPUIndent(dpufile);
  std::string vid = AllocVarID(op->loop_var.get());
  CHECK(is_zero(op->min));
  stream << "for (";
	dpufile << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = 0; "
            << vid << " < " << extent
            << "; ++" << vid << ") {\n";
	dpufile << ' ' << vid << " = 0; "
            << vid << " < " << extent
            << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
	PrintDPUIndent(dpufile);
  stream << "}\n";
	dpufile << "}\n";
}

void CodeGenDPUHost::VisitStmt_(const StoreNode* op) {
  DataType t = op->value.dtype();
  if (t.lanes() == 1) {
    std::string value = this->PrintExpr(op->value);
    std::string ref  = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();
		PrintDPUIndent(dpufile);
    stream << ref << " = " << value << ";\n";
		dpufile << ref << " = " << value << ";\n";
  } else {
    CHECK(is_one(op->predicate))
        << "Predicated store is not supported";
    PrimExpr base;
    if (GetRamp1Base(op->index, t.lanes(), &base)) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer_var.get(), t, base, value);
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements seperately
      std::string index = SSAGetID(PrintExpr(op->index), op->index.dtype());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.dtype());
      std::string vid = GetVarID(op->buffer_var.get());
      for (int i = 0; i < t.lanes(); ++i) {
        this->PrintIndent();
				this->PrintDPUIndent(dpufile);
        DataType elem_type = t.element_of();
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          stream << "((";
					dpufile << "((";
          if (op->buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, stream);
              stream << ' ';
						  dpufile << ' ';
            }
          }
          PrintType(elem_type, stream);
          stream << "*)" << vid << ')';
					dpufile << "*)" << vid << ')';
        } else {
          stream << vid;
					dpufile << vid;
        }
        stream << '[';
				dpufile << '[';
        PrintVecElemLoad(index, op->index.dtype(), i, stream);
        stream << "] = ";
				dpufile << "] = ";
        PrintVecElemLoad(value, op->value.dtype(), i, stream);
        stream << ";\n";
				dpufile << ";\n";
      }
      EndScope(vec_scope);
    }
  }
}



void CodeGenDPUHost::VisitStmt_(const IfThenElseNode* op) {
  std::string cond = PrintExpr(op->condition);
  //PrintIndent();
  if (cond[0] == '(' && cond[cond.length() - 1] == ')') {
  //  stream << "if " << cond << " {\n";
  } else {
  //  stream << "if (" << cond << ") {\n";
  }
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);

  if (op->else_case.defined()) {
  //  PrintIndent();
  //  stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case);
    this->EndScope(else_scope);
  }
  //PrintIndent();
  //stream << "}\n";
}


void CodeGenDPUHost::VisitStmt_(const AssertStmtNode *op) { // NOLINT(*)
  if (emit_asserts_) {
    std::string cond = PrintExpr(op->condition);
    PrintIndent();
	  PrintDPUIndent(dpufile);
    stream << "if (!(" << cond << ")) {\n";
		dpufile << "if (!(" << cond << ")) {\n";
    int assert_if_scope = this->BeginScope();
    PrintIndent();
		PrintDPUIndent(dpufile);
    stream << "TVMAPISetLastError(\"" << op->message.as<StringImmNode>()->value << "\");\n";
    dpufile << "TVMAPISetLastError(\"" << op->message.as<StringImmNode>()->value << "\");\n";
		PrintIndent();
		PrintDPUIndent(dpufile);
    stream << "return -1;\n";
		dpufile << "return -1;\n";
    this->EndScope(assert_if_scope);
    PrintIndent();
		PrintDPUIndent(dpufile);
    stream << "}\n";
		dpufile << "}\n";
  }
  this->PrintStmt(op->body);
}

void CodeGenDPUHost::VisitExpr_(const MinNode *op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, "<", os, dpufile);
}

void CodeGenDPUHost::VisitExpr_(const MaxNode *op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, ">", os, dpufile);
}

void CodeGenDPUHost::PrintSSAAssign(
    const std::string& target, const std::string& src, DataType t) {
  PrintType(t, stream);
  stream << ' ' << target << " = ";
	dpufile << ' ' << target << " = ";
  if (src.length() > 3 &&
      src[0] == '(' && src[src.length() - 1] == ')') {
    stream << src.substr(1, src.length() - 2);
		dpufile << src.substr(1, src.length() - 2);
  } else {
    stream << src;
		dpufile << src;
  }
  stream << ";\n";
	dpufile << ";\n";
}


template <typename T>
inline void CodeGenDPUHost::PrintTernaryCondExpr(const T* op,
                                           const char* compare,
                                           std::ostream& os,
                                           std::fstream& dpufile) {  // NOLINT(*)
  std::ostringstream temp_a;
  VisitExpr(op->a, temp_a);
	PrintDPUIndent(dpufile);
  std::string a_id = SSAGetID(temp_a.str(), op->a.dtype());
  std::ostringstream temp_b;
  VisitExpr(op->b, temp_b);
	PrintDPUIndent(dpufile);
  std::string b_id = SSAGetID(temp_b.str(), op->b.dtype());

  os << "((" << a_id << ") " << compare << " (" << b_id << ") "
     << "? (" << a_id << ") : (" << b_id << "))";
	//dpufile << "((" << a_id << ") " << compare << " (" << b_id << ") "
  //   << "? (" << a_id << ") : (" << b_id << "))";
}

runtime::Module BuildDPUHost(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  bool emit_asserts = false;
  CodeGenDPUHost cg;
  cg.Init(output_ssa, emit_asserts);
  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c");
}

TVM_REGISTER_GLOBAL("codegen.build_dpu")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildDPUHost(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
