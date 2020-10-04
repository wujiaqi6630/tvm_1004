for (size_t i = 0; i < f->args.size(); ++i) {
  //for (size_t i = 0; i < argsList_.size(); ++i) {
    Var v = f->args[i];
    //Var v = argsList_[i];
    std::string vid = AllocVarID(v.get());
    //std::string vid = v.get()->name_hint;
    //if (i != 0) stream << ", ";
    //if (v.dtype().is_handle()) {
      //auto it = alloc_storage_scope_.find(v.get());
      //if (it != alloc_storage_scope_.end()) {
      //  PrintStorageScope(it->second, stream);
      //}
      //stream << ' ';


      //if (handle_data_type_.count(v.get())) {
      //  PrintType(handle_data_type_.at(v.get()), stream);
      //} else {
      //  stream << "void";
      //}
      //stream << "*";

      //if (f->is_restricted && restrict_keyword_.length() != 0) {
      //  stream << ' ' << restrict_keyword_;
      //}
   // } else {
     // PrintType(v.dtype(), stream);
    //}
   // stream << ' ' << vid;
  }
  int len = (tvm::ir::args_list).size();
  std::cout<< "arg_list size is : "<< len << std::endl;
  /*******************wjq********************//*
  for (size_t i = 0; i < argsList_.size(); ++i) {
    Var v = argsList_[i];
    //can not use "AllocVarID(v.get())" here
    std::string vid = v.get()->name_hint;
    if (i != 0) stream << ", ";
    if (v.dtype().is_handle()) {
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }
      if (i != 0) stream << ' ';
      if (handle_data_type_.count(v.get())) {
        PrintType(handle_data_type_.at(v.get()), stream);
      }
else {
          stream << "void";
      }
      stream << "*";
    } 
    else {
        PrintType(v.dtype(), stream);
    }
    stream << ' ' << vid;
  }
  *//*******************wjq*******************/

