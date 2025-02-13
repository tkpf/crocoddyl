///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/frame-xaxis-orientation.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualFrameXAxisOrientation() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelFrameXAxisOrientation> >();

  bp::class_<ResidualModelFrameXAxisOrientation, bp::bases<ResidualModelAbstract> >(
      "ResidualModelFrameXAxisOrientation",
      "This residual function is defined as r = R - Rref, with R and Rref as "
      "the current and reference\n"
      "frame rotations, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
               Eigen::Matrix3d, std::size_t>(
          bp::args("self", "state", "id", "Rref", "nu"),
          "Initialize the frame rotation residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param Rref: reference frame rotation\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
                    Eigen::Matrix3d>(
          bp::args("self", "state", "id", "Rref"),
          "Initialize the frame rotation residual model.\n\n"
          "The default nu value is obtained from model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param Rref: reference frame rotation"))
      .def<void (ResidualModelFrameXAxisOrientation::*)(
          const boost::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelFrameXAxisOrientation::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the frame rotation residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelFrameXAxisOrientation::*)(
          const boost::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelFrameXAxisOrientation::*)(
          const boost::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelFrameXAxisOrientation::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the frame rotation residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelFrameXAxisOrientation::*)(
          const boost::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ResidualModelFrameXAxisOrientation::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame rotation residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for the frame rotation residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelFrameXAxisOrientation::get_id,
                    &ResidualModelFrameXAxisOrientation::set_id, "reference frame id")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelFrameXAxisOrientation::get_reference,
                            bp::return_internal_reference<>()),
          &ResidualModelFrameXAxisOrientation::set_reference,
          "reference frame rotation")
      .def(CopyableVisitor<ResidualModelFrameXAxisOrientation>());

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataFrameXAxisOrientation> >();

  bp::class_<ResidualDataFrameXAxisOrientation, bp::bases<ResidualDataAbstract> >(
      "ResidualDataFrameXAxisOrientation", "Data for frame rotation residual.\n\n",
      bp::init<ResidualModelFrameXAxisOrientation*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame rotation residual data.\n\n"
          ":param model: frame rotation residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataFrameXAxisOrientation::pinocchio,
                                    bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("r",
                    bp::make_getter(&ResidualDataFrameXAxisOrientation::r,
                                    bp::return_internal_reference<>()),
                    "residual residual")
      .add_property("rRf",
                    bp::make_getter(&ResidualDataFrameXAxisOrientation::rRf,
                                    bp::return_internal_reference<>()),
                    "rotation error of the frame")
      .add_property("rJf",
                    bp::make_getter(&ResidualDataFrameXAxisOrientation::rJf,
                                    bp::return_internal_reference<>()),
                    "error Jacobian of the frame")
      .add_property("fJf",
                    bp::make_getter(&ResidualDataFrameXAxisOrientation::fJf,
                                    bp::return_internal_reference<>()),
                    "local Jacobian of the frame")
      .def(CopyableVisitor<ResidualDataFrameXAxisOrientation>());
}

}  // namespace python
}  // namespace crocoddyl
