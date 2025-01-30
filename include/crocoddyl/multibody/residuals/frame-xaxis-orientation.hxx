///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// tkpf
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/math/quaternion.hpp>
#include <Eigen/Geometry>


#include "crocoddyl/multibody/residuals/frame-xaxis-orientation.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelFrameXAxisOrientationTpl<Scalar>::ResidualModelFrameXAxisOrientationTpl(   // TODO constructor with only x-axis orientation, no need to extract
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Matrix3s& Rref, const std::size_t nu)
    : Base(state, 3, nu, true, false, false),
      id_(id),
      Rref_(Rref),
      oRf_inv_(Rref.transpose()),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelFrameXAxisOrientationTpl<Scalar>::ResidualModelFrameXAxisOrientationTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Matrix3s& Rref)
    : Base(state, 3, true, false, false),
      id_(id),
      Rref_(Rref),
      oRf_inv_(Rref.transpose()),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelFrameXAxisOrientationTpl<Scalar>::~ResidualModelFrameXAxisOrientationTpl() {}

template <typename Scalar>
void ResidualModelFrameXAxisOrientationTpl<Scalar>::calc(
    const boost::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame rotation w.r.t. the reference frame
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);

  // Rotation around the x-Axis (1, 0, 0) has the form qx​=(sinθ/2​,0,0, cosθ/2​) | convention (x, y, z ,w); see https://github.com/stack-of-tasks/pinocchio/issues/16#issuecomment-94799789
  // We only care about x-Axis alignment, hence we  throw away y,z quaterion element
  // Given a rotation matrix instead of a quaternion we have the following: R = [[1, 0, 0], [0, a, b], [0, c, d]]; if we use rotation matrix and setting some elemnts to 0 and 1 we come into problems facing no orthogonality
  Matrix3s rRf;
  Eigen::Quaterniond q_rRf;
  rRf.noalias() = oRf_inv_ * d->pinocchio->oMf[id_].rotation();
  q_rRf = Eigen::Quaterniond(rRf);
  q_rRf.y() = 0
  q_rRf.z() = 0
  q_rRf.normalize()
  // d->x_aligned_rRf.noalias() = q_rRf.toRotationMatrix()
  d->rRf.noalias() = q_rRf.toRotationMatrix()   // TODO rename?   // TODO we need to adapt residualdata in frame-axis-orientation.hpp

  data->r = pinocchio::log3(d->x_aligned_rRf);
}

template <typename Scalar>
void ResidualModelFrameXAxisOrientationTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame Jacobian at the error point
  pinocchio::Jlog3(d->rRf, d->rJf);
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_,
                              pinocchio::LOCAL, d->fJf);

  // Compute the derivatives of the frame rotation
  const std::size_t nv = state_->get_nv();
  data->Rx.leftCols(nv).noalias() = d->rJf * d->fJf.template bottomRows<3>();
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelFrameXAxisOrientationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void ResidualModelFrameXAxisOrientationTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  typename pinocchio::SE3Tpl<Scalar>::Quaternion qref;
  pinocchio::quaternion::assignQuaternion(qref, Rref_);
  os << "ResidualModelFrameXAxisOrientation {frame=" << pin_model_->frames[id_].name
     << ", qref=" << qref.coeffs().transpose().format(fmt) << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFrameXAxisOrientationTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Matrix3s&
ResidualModelFrameXAxisOrientationTpl<Scalar>::get_reference() const {
  return Rref_;
}

template <typename Scalar>
void ResidualModelFrameXAxisOrientationTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFrameXAxisOrientationTpl<Scalar>::set_reference(
    const Matrix3s& rotation) {
  Rref_ = rotation;
  oRf_inv_ = rotation.transpose();
}

}  // namespace crocoddyl
