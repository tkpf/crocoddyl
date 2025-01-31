///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh and tkpf
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>

#include "crocoddyl/multibody/residuals/frame-push-off.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelFramePushOffTpl<Scalar>::ResidualModelFramePushOffTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Vector3s& xref, const Scalar dist, const std::size_t nu)
    : Base(state, 3, nu, true, false, false),
      id_(id),
      xref_(xref),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelFramePushOffTpl<Scalar>::ResidualModelFramePushOffTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Vector3s& xref, const Scalar dist)
    : Base(state, 3, true, false, false),
      id_(id),
      xref_(xref),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelFramePushOffTpl<Scalar>::~ResidualModelFramePushOffTpl() {}

template <typename Scalar>
void ResidualModelFramePushOffTpl<Scalar>::calc(
    const boost::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  // Compute the frame translation w.r.t. the reference frame
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  data->r = dist_ - (d->pinocchio->oMf[id_].translation() - xref_);
}

template <typename Scalar>
void ResidualModelFramePushOffTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame translation
  const std::size_t nv = state_->get_nv();
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_,
                              pinocchio::LOCAL, d->fJf);
  d->Rx.leftCols(nv).noalias() =
      - (d->pinocchio->oMf[id_].rotation() * d->fJf.template topRows<3>()); // minus for push off // TODO is this correct?
  ;
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelFramePushOffTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void ResidualModelFramePushOffTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "ResidualModelFramePushOff {frame=" << pin_model_->frames[id_].name
     << ", tref=" << xref_.transpose().format(fmt) << "dist=" << dist_ << "}"; // TODO is this correct
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFramePushOffTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s&
ResidualModelFramePushOffTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
pinocchio::FrameIndex<Scalar> ResidualModelFramePushOffTpl::get_distance() const {
  return dist_;
}

template <typename Scalar>
void ResidualModelFramePushOffTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFramePushOffTpl<Scalar>::set_reference(
    const Vector3s& translation) {
  xref_ = translation;
}

template <typename Scalar>
void ResidualModelFramePushOffTpl<Scalar>::set_distance(
    const pinocchio::FrameIndex dist) {
  dist_ = dist;
}

}  // namespace crocoddyl
