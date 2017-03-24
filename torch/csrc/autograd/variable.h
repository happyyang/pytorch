#pragma once

#include <mutex>
#include <memory>
#include <functional>
#include <THPP/THPP.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable_version.h"
#include "torch/csrc/Types.h"

namespace torch { namespace autograd {

struct Variable : std::enable_shared_from_this<Variable> {

  struct SavedVariable {
    SavedVariable()
      : data()
      , version()
      , expected_version(-1) {}

    SavedVariable(const Variable& variable)
      : data(variable.data->clone_shallow())
      , grad_fn(variable.grad_fn)
      , version(std::move(variable.version_counter->new_saved_ref()))
      , requires_grad(variable.requires_grad)
      , is_volatile(false)
      , expected_version(**variable.version_counter) {}

    std::unique_ptr<thpp::Tensor> data;
    std::shared_ptr<Function> grad_fn;
    std::unique_ptr<VariableVersion> version;
    bool requires_grad;
    bool is_volatile;
    int expected_version;

    std::shared_ptr<Variable> unpack();

    std::unique_ptr<thpp::Tensor> unpack_data() {
      auto var = unpack();
      return var ? std::move(var->data) : nullptr;
    }
  };

  // WARNING: this registers the Variable as a new output
  Variable(
      std::unique_ptr<thpp::Tensor> data,
      std::shared_ptr<Function> grad_fn);

  Variable(
      std::unique_ptr<thpp::Tensor> data,
      bool requires_grad,
      bool is_volatile);

  std::shared_ptr<Function> get_grad_accumulator();

  inline SavedVariable save() {
    return SavedVariable(*this);
  }

  static inline SavedVariable save_opt(Variable* var) {
    return var ? var->save() : SavedVariable();
  }

  static inline std::shared_ptr<Variable> of(std::unique_ptr<thpp::Tensor> data, bool is_volatile=false) {
    if (!data) {
      return std::shared_ptr<Variable>();
    }
    return std::make_shared<Variable>(std::move(data), false, is_volatile);
  }

  std::unique_ptr<thpp::Tensor> data;
  std::shared_ptr<Function> grad_fn;
  std::shared_ptr<Variable> grad;
  std::unique_ptr<VariableVersion> version_counter;
  std::vector<std::shared_ptr<FunctionPreHook>> hooks;
  std::weak_ptr<Function> grad_accumulator;
  std::mutex grad_accumulator_lock;
  bool requires_grad;
  bool is_volatile;
  int output_nr;
  PyObject *pyobj;  // weak reference
};

using SavedVariable = Variable::SavedVariable;

}} // namespace torch::autograd
