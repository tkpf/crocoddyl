///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/frame-push-off.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualFramePushOff() {
  bp::register_ptr_to_python<
      boost::shared_ptr<ResidualModelFramePushOff> >();

  bp::class_<ResidualModelFramePushOff, bp::bases<ResidualModelAbstract> >(
      "ResidualModelFramePushOff",
      "This residual function defines the the frame push off as "
      "r = dist - (t - tref), with t and tref as the\n"
      "current and reference frame translations, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
               Eigen::Vector3d, std::size_t>(
          bp::args("self", "state", "id", "xref", "dist", "nu"),
          "Initialize the frame translation residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param xref: reference frame translation\n"
          ":param dist: minimal distance to be enforced\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
                    Eigen::Vector3d>(
          bp::args("self", "state", "id", "xref", "dist"),
          "Initialize the frame translation residual model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param xref: reference frame translation"
          ":param dist: minimal distance to be enforced\n"))
      .def<void (ResidualModelFramePushOff::*)(
          const boost::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelFramePushOff::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the frame translation residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelFramePushOff::*)(
          const boost::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelFramePushOff::*)(
          const boost::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelFramePushOff::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the frame translation residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelFramePushOff::*)(
          const boost::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ResidualModelFramePushOff::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame translation residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for the frame translation residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelFramePushOff::get_id,
                    &ResidualModelFramePushOff::set_id,
                    "reference frame id")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelFramePushOff::get_reference,
                            bp::return_internal_reference<>()),
          &ResidualModelFramePushOff::set_reference,
          "reference frame translation")
       .add_property(
          "distance",
          bp::make_function(&ResidualModelFramePushOff::get_distance,
                            bp::return_internal_reference<>()),
          &ResidualModelFramePushOff::set_reference,
          "minimal distance to be enforced")
      .def(CopyableVisitor<ResidualModelFramePushOff>());

  bp::register_ptr_to_python<
      boost::shared_ptr<ResidualDataFramePushOff> >();

  bp::class_<ResidualDataFramePushOff, bp::bases<ResidualDataAbstract> >(
      "ResidualDataFramePushOff",
      "Data for frame translation residual.\n\n",
      bp::init<ResidualModelFramePushOff*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame translation residual data.\n\n"
          ":param model: frame translation residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataFramePushOff::pinocchio,
                                    bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("fJf",
                    bp::make_getter(&ResidualDataFramePushOff::fJf,
                                    bp::return_internal_reference<>()),
                    "local Jacobian of the frame")
      .def(CopyableVisitor<ResidualDataFramePushOff>());
}

}  // namespace python
}  // namespace crocoddyl
