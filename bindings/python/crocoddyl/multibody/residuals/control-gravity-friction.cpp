///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/control-gravity-friction.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualControlGravFric() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelControlGravFric> >();

  bp::class_<ResidualModelControlGravFric, bp::bases<ResidualModelAbstract> >(
      "ResidualModelControlGravFric",
      "This residual function is defined as r = a(u) - g(q), where a(u)\n"
      "is the actuated torque; and q, g(q) are the generalized position\n"
      "and gravity vector, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>,
            const Eigen::VectorXd&,
            const Eigen::VectorXd&,
            std::size_t>(
          bp::args("self", "state", "v_friction_coeff", "s_friction_coeff", "nu"),
          "Initialize the control-gravity-friction residual model.\n\n"
          ":param state: state description\n"
          ":param v_friction_coeff: viscous friction coefficients\n"
          ":param s_friction_coeff: static friction coefficients\n"
          ":param nu: dimension of the control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>,
            const Eigen::VectorXd&,
            const Eigen::VectorXd&>(
          bp::args("self", "state", "v_friction_coeff", "s_friction_coeff"),
          "Initialize the control-gravity-friction residual model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param v_friction_coeff: viscous friction coefficients\n"
          ":param s_friction_coeff: static friction coefficients"))
      .def<void (ResidualModelControlGravFric::*)(
          const boost::shared_ptr<ResidualDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &ResidualModelControlGravFric::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the control residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelControlGravFric::*)(
          const boost::shared_ptr<ResidualDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelControlGravFric::*)(
          const boost::shared_ptr<ResidualDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &ResidualModelControlGravFric::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the control residual.\n\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelControlGravFric::*)(
          const boost::shared_ptr<ResidualDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ResidualModelControlGravFric::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the control residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This "
           "function\n"
           "returns the allocated data for the control gravity residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .def(CopyableVisitor<ResidualModelControlGravFric>());

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataControlGravFric> >();

  bp::class_<ResidualDataControlGravFric, bp::bases<ResidualDataAbstract> >(
      "ResidualDataControlGravFric", "Data for control gravity residual.\n\n",
      bp::init<ResidualModelControlGravFric *, DataCollectorAbstract *>(
          bp::args("self", "model", "data"),
          "Create control gravity residual data.\n\n"
          ":param model: control gravity residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataControlGravFric::pinocchio),
                    "Pinocchio data used for internal computations")
      .add_property("actuation",
                    bp::make_getter(&ResidualDataControlGravFric::actuation,
                                    bp::return_internal_reference<>()),
                    "actuation model")
      .def(CopyableVisitor<ResidualDataControlGravFric>());
}

}  // namespace python
}  // namespace crocoddyl
