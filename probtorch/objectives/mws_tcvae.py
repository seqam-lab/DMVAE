def elbo(q, p, rec, latents=None, sample_dim=None, batch_dim=None, beta=[1.0,1.0,1.0],
         bias=1.0):
    reconst_loss = rec.loss

    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_probability(q, p, latents, sample_dim, batch_dim, bias)
    kl = beta[0] * (log_q_zCx - log_qz) +  beta[1] * (log_qz - log_prod_qzi) + beta[2] * (log_prod_qzi - log_pz)

    return (reconst_loss.mean(), kl.mean())

def _get_probability(q, p, latents, sample_dim, batch_dim, bias):
    log_pz = p.log_joint(sample_dim, batch_dim, latents) # =  p['private'].log_prob.sum(2) + p['shared'].log_prob. # sum over all latent dim(priv +_ shared). size = (1,batch_size)
    log_q_zCx = q.log_joint(sample_dim, batch_dim, latents) # = q['private'].log_prob.sum(2) + q['shared'].log_prob
    log_joint_qz_marginal, _, log_prod_qzi = q.log_batch_marginal(sample_dim, batch_dim, latents, bias=bias)
    return log_pz, log_joint_qz_marginal, log_prod_qzi, log_q_zCx

