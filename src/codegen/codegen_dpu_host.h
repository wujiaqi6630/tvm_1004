
/*!
 * \file codegen_dpu_host.h
 */
 
#ifndef TVM_CODEGEN_CODEGEN_DPU_HOST_H_
#define TVM_CODEGEN_CODEGEN_DPU_HOST_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include <vector>
#include "codegen_c.h"
#include "iostream"
#include<fstream>
#include<iomanip>

namespace tvm {
namespace codegen {

class CodeGenDPUHost final : public CodeGenC {
 public:
  CodeGenDPUHost();
  void Init(bool output_ssa, bool emit_asserts);
  void AddFunction(LoweredFunc f);
  std::string Finish();

  void PrintType(DataType t, std::ostream& os) final; // NOLINT(*)
  //PrintConst(const IntImmNode* op, std::ostream& os, CodeGenC* p)

	void PrintVecStore(const VarNode* buffer,
                     DataType t, PrimExpr base,
                     const std::string& value) final;

	void PrintVecElemLoad(const std::string& vec,
											 DataType t, int i,
											 std::ostream& os) final;
                     
  // overload visitor functions
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const CallNode *op, std::ostream& os) final; // NOLINT(*)
  // overload min and max to use the ternary operator, so we don't rely on the
  // standard library implementations
  void VisitExpr_(const MinNode *op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const MaxNode *op, std::ostream& os) final;  // NOLINT(*)

  /**********wjq**********/
	//void VisitExpr_(const LoadNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitStmt_(const LetStmtNode* op) final;   // NOLINT(*)
  void VisitStmt_(const AttrStmtNode* op) final;   // NOLINT(*)
  void VisitStmt_(const ForNode* op) final;   // NOLINT(*)
  void VisitStmt_(const IfThenElseNode* op) final;
  void VisitStmt_(const AllocateNode* op) final;
	void VisitStmt_(const StoreNode* op) final;
  /**********wjq**********/
	
  void VisitStmt_(const AssertStmtNode *op) final; // NOLINT(*)

	// override
  void PrintSSAAssign(
      const std::string& target, const std::string& src, DataType t) final;

 private:
  std::string module_name_;
  /***wjq***/
  bool hasPrintedArgsList;
  std::string functionName;
  Array<Var> args_list;
  std::unordered_map<std::string, bool> argsListMap;
  std::vector<std::string> argsVec;

  std::string dpuCodeLibPath;
	std::fstream dpufile;
  /***wjq***/
  /*! \brief whether to emit asserts in the resulting C code */
  
  bool emit_asserts_;

  void PrintGetFuncFromBackend(const std::string& func_name, const std::string& packed_func_name);
  void PrintFuncCall(const std::string& packed_func_name, int num_args);

  /*!
   * \brief Print ternary conditional operator implementing binary `op`
   * Forces the operands to be in SSA form.
   * \param op binary operator being expressed
   * \param compare string representation of comparison operator
   * \param os stream reference to print into
   */
  template <typename T>
  inline void PrintTernaryCondExpr(const T* op,
                                   const char* compare,
                                   std::ostream& os,
                                   std::fstream& dpufile);  // NOLINT(*)
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_C_HOST_H_
