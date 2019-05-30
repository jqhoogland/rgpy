import tensorflow as tf

n_visible = 5
n_hidden = 2


# shape ?,5
sample_v_0 = tf.reshape(tf.constant([1, 2, 3, 4, 5, 2, 4, 6, 8, 10, 1, 1, 1, 1, 1], dtype=tf.float32), [3, 5]) / tf.constant(1000, dtype=tf.float32)
# shape ?,2
sample_h_0 = tf.reshape(tf.constant([1, 2, 0.5, 1, 1, 1], dtype=tf.float32), [3, 2])


W = tf.Variable(tf.random.uniform([n_visible, n_hidden]),
                dtype=tf.float32)

hbias = tf.Variable(tf.random.uniform([n_hidden]),
                    dtype=tf.float32)

local_field_lambda_j=lambda v: (tf.matmul(v, W)) + hbias
energy_lambda_j=lambda v, h: local_field_lambda_j(v) * h
energy_lambda_of_vh=lambda v, h: tf.reduce_sum(energy_lambda_j(v, h), axis=1)
energy_lambda_of_v=lambda v: tf.reduce_sum(tf.log(1 + tf.exp(local_field_lambda_j(v))), axis=1)

# TARGET [?,2]
h_exp_0 = local_field_lambda_j(sample_v_0)

# TARGET [?,2]
elj_0 = energy_lambda_j(sample_v_0, sample_h_0)

tmp0=1 + tf.exp(local_field_lambda_j(sample_v_0))
tmp1=tf.log(tmp0)
tmp2=tf.reduce_sum(local_field_lambda_j(sample_v_0), axis=1)

# TARGET [?]
elvh_00=energy_lambda_of_vh(sample_v_0, sample_h_0)

# TARGET [?]
elv_0=energy_lambda_of_v(sample_v_0)

init_op=tf.global_variables_initializer()



delta_of_vh=lambda v, h: [h, tf.einsum('ij,ik->ijk', v, h)]
delta_of_vh_t=lambda v, h: [h, tf.expand_dims(h, 1) * tf.expand_dims(v, 2)]
# TODO: figure out which of these two options is faster. My guess is it's the second, but that's just my guess.
delta_of_vh_avg = lambda v, h: [tf.reduce_sum(h, axis= 0), tf.matmul(v, h, transpose_a=True)]

deltas_0 = delta_of_vh(sample_v_0, sample_h_0)
deltas_1 = delta_of_vh_avg(sample_v_0, sample_h_0)
deltas_2 = delta_of_vh_t(sample_v_0, sample_h_0)


h_eff_0 = sample_h_0 - h_exp_0
geplvh_0 = delta_of_vh_t(sample_v_0, h_eff_0)

with tf.Session() as sess:
    sess.run(init_op)
    ([sample_v_0_, sample_h_0_]) = sess.run([sample_v_0, sample_h_0])

    # Check whether simple energy calculations produce the correct results
    (
        [h_exp_0_, elj_0_, elvh_00_, elv_0]
    ) = sess.run(
        [h_exp_0, elj_0, elvh_00, elv_0]
    )

    # Check whether computing of marginal energy occurs in correct order
    (
        [tmp0_, tmp1_, tmp2_]
    ) = sess.run(
        [tmp0, tmp1, tmp2]
    )

    # Check whether the deltas (i.e. h_j and v_i *h_j) are correctly computed
    (
    [deltas_0_, deltas_1_, deltas_2_, geplvh_0_]
    )p = sess.run(
        [deltas_0, deltas_1, deltas_2, geplvh_0]
    )

    print("Samples are Visible:\n{}\nHidden\n{}"
          .format(sample_v_0_, sample_h_0_))

    print("Local field:\n{}\nLocal energies:\n{}\nTotal energy:\n{}\nMarginal energy (of v):\n{}"
          .format(h_exp_0_, elj_0_, elvh_00_, elv_0))

    print("debugging:\n{}\n{}\n{}".format(tmp0_, tmp1_, tmp2_))
    print("Correct ordering:", (tmp2_ != elv_0))

    print("Deltas:\n{}\n{}\n\n{} with shapes{}, {}, and {}"
          .format(deltas_0_,deltas_1_, deltas_2_, [d.shape for d in deltas_0_],
                  [d.shape for d in deltas_1_], [d.shape for d in deltas_2_]))

    print("Grad energy psi lambda of vh: \n{}".format(geplvh_0_))
