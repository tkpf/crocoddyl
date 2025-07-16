///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2022, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace crocoddyl {

template <typename Scalar>
ResidualModelControlGravFricTpl<Scalar>::ResidualModelControlGravFricTpl(
    boost::shared_ptr<StateMultibody> state, const VectorXs& vf_coeff,
    const VectorXs& cf_coeff, const std::size_t nu)
    : Base(state, state->get_nv(), nu, true, false),
      pin_model_(*state->get_pinocchio()),
      viscous_friction_coeff_(vf_coeff),
      static_friction_coeff_(cf_coeff) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add "
                    "this residual function");
  }
}

template <typename Scalar>
ResidualModelControlGravFricTpl<Scalar>::ResidualModelControlGravFricTpl(
    boost::shared_ptr<StateMultibody> state, const VectorXs& vf_coeff,
    const VectorXs& cf_coeff)
    : Base(state, state->get_nv(), state->get_nv(), true, false),
      pin_model_(*state->get_pinocchio()),
      viscous_friction_coeff_(vf_coeff),
      static_friction_coeff_(cf_coeff) {}

template <typename Scalar>
ResidualModelControlGravFricTpl<Scalar>::~ResidualModelControlGravFricTpl() {}

template <typename Scalar>
void ResidualModelControlGravFricTpl<Scalar>::calc(
    const boost::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());
  // parse to array datatype, so that we can use element-wise operations
  d->friction_.noalias() = (viscous_friction_coeff_ * v.array() + static_friction_coeff_ * v.array().sign()).matrix();


  data->r = d->actuation->tau -
            pinocchio::computeGeneralizedGravity(pin_model_, d->pinocchio, q) -
            d->friction_;

  std::cout << "re-calculated friction:\n" << d->friction_ << std::endl;
  std::cout << "static friction:\n" << static_friction_coeff_ * v.array().sign() << std::endl;
  std::cout << "viscous friction:\n" << viscous_friction_coeff_ * v.array() << std::endl;
}

template <typename Scalar>
void ResidualModelControlGravFricTpl<Scalar>::calc(
    const boost::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x) {
  Data *d = static_cast<Data *>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());
  // parse to array datatype, so that we can use element-wise operations
  d->friction_.noalias() = (viscous_friction_coeff_ * v.array() + static_friction_coeff_ * v.array().sign()).matrix();

  data->r = -pinocchio::computeGeneralizedGravity(pin_model_, d->pinocchio, q) - d->friction_;
  std::cout << "re-calculated friction:\n" << d->friction_ << std::endl;
  std::cout << "static friction:\n" << static_friction_coeff_ * v.array().sign() << std::endl;
  std::cout << "viscous friction:\n" << viscous_friction_coeff_ * v.array() << std::endl;
}

template <typename Scalar>
void ResidualModelControlGravFricTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  // Compute the derivatives of the residual residual
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq =
      data->Rx.leftCols(state_->get_nv());
  pinocchio::computeGeneralizedGravityDerivatives(pin_model_, d->pinocchio, q,
                                                  Rq);
  Rq *= -1;

  //  Compute the derivatives of the joint velocity (only dependent on friction)
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());
  Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rv =
      data->Rx.rightCols(state_->get_nv());

  // NOTE: modelling discrete friciton sign function with: smooth_sign = (k * qdot.array()).tanh(); -> derivate:
  ArrayXs dsmooth_dqdot = k_friction_smoothing * ((k_friction_smoothing * v.array()).cosh().square()).inverse();
  Rv = (viscous_friction_coeff_ + static_friction_coeff_ * dsmooth_dqdot).matrix().asDiagonal();
  
  Rv *= -1;

  std::cout << "Rq:\n" << Rq << std::endl;
  std::cout << "Rv:\n" << Rv << std::endl;

  data->Ru = d->actuation->dtau_du;
}

template <typename Scalar>
void ResidualModelControlGravFricTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x) {
  Data *d = static_cast<Data *>(data.get());

  // Compute the derivatives of the residual residual
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq =
      data->Rx.leftCols(state_->get_nv());
  pinocchio::computeGeneralizedGravityDerivatives(pin_model_, d->pinocchio, q,
                                                  Rq);
  Rq *= -1;

  //  Compute the derivatives of the joint velocity (only dependent on friction)
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());
  Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rv =
      data->Rx.rightCols(state_->get_nv());

  // NOTE: modelling discrete friciton sign function with: smooth_sign = (k * qdot.array()).tanh(); -> derivate:
  ArrayXs dsmooth_dqdot = k_friction_smoothing * ((k_friction_smoothing * v.array()).cosh().square()).inverse();
  Rv = (viscous_friction_coeff_ + static_friction_coeff_ * dsmooth_dqdot).matrix().asDiagonal();
  
  Rv *= -1;

  std::cerr << "Rq:\n" << Rq << std::endl;
  std::cerr << "Rv:\n" << Rv << std::endl;

}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelControlGravFricTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void ResidualModelControlGravFricTpl<Scalar>::print(std::ostream &os) const {
  os << "ResidualModelControlGravFric";
}

}  // namespace crocoddyl
