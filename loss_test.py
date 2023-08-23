def custom_loss(y_true, y_pred):

    import tensorflow.keras.backend as K
    import tensorflow as tf

    px_truth = K.flatten(y_true[:, 0])
    py_truth = K.flatten(y_true[:, 1])
    px_pred = K.flatten(y_pred[:, 0])
    py_pred = K.flatten(y_pred[:, 1])

    pt_truth = K.sqrt(px_truth*px_truth + py_truth*py_truth)
    pt_pred = K.sqrt(px_pred*px_pred + py_pred*py_pred)

    upar_pred = pt_pred - pt_truth
    pt_cut = pt_truth > 0.
    pt_pred_filtered = tf.boolean_mask(pt_pred, pt_cut)
    upar_pred = tf.boolean_mask(upar_pred, pt_cut)
    pt_truth_filtered = tf.boolean_mask(pt_truth, pt_cut)

    filter_bin0 = tf.logical_and(pt_truth_filtered > 50.,  pt_truth_filtered < 100.)
    filter_bin1 = tf.logical_and(pt_truth_filtered > 100., pt_truth_filtered < 200.)
    filter_bin2 = tf.logical_and(pt_truth_filtered > 200., pt_truth_filtered < 300.)
    filter_bin3 = tf.logical_and(pt_truth_filtered > 300., pt_truth_filtered < 400.)
    filter_bin4 = pt_truth_filtered > 400.

    upar_pred_pos_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred > 0.))
    upar_pred_neg_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred < 0.))
    upar_pred_pos_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred > 0.))
    upar_pred_neg_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred < 0.))
    upar_pred_pos_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred > 0.))
    upar_pred_neg_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred < 0.))
    upar_pred_pos_bin3 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin3, upar_pred > 0.))
    upar_pred_neg_bin3 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin3, upar_pred < 0.))
    upar_pred_pos_bin4 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin4, upar_pred > 0.))
    upar_pred_neg_bin4 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin4, upar_pred < 0.))
    #upar_pred_pos_bin5 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin5, upar_pred > 0.))
    #upar_pred_neg_bin5 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin5, upar_pred < 0.))
    norm = tf.reduce_sum(pt_truth_filtered)
    dev = tf.abs(tf.reduce_sum(upar_pred_pos_bin0) + tf.reduce_sum(upar_pred_neg_bin0))
    dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin1) + tf.reduce_sum(upar_pred_neg_bin1))
    dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin2) + tf.reduce_sum(upar_pred_neg_bin2))
    dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin3) + tf.reduce_sum(upar_pred_neg_bin3))
    dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin4) + tf.reduce_sum(upar_pred_neg_bin4))

    dev /= norm

    def huber_loss(y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic_region = 0.5 * tf.square(abs_error)
        linear_region = delta * abs_error - 0.5 * tf.square(delta)
        loss = tf.where(abs_error <= delta, quadratic_region, linear_region)
        return tf.reduce_mean(loss)

    def quantile_loss(y_true, y_pred, tau):
        error = y_true - y_pred
        loss = tf.where(error > 0, tau * error, (tau - 1) * error)
        return tf.reduce_mean(loss)

    delta = 1.0  # Huber loss delta
    tau_25 = 0.25  # 25% quantile
    tau_75 = 0.75  # 75% quantile
 
    def calculate_quantile(predictions, quantile):
        sorted_predictions = tf.sort(predictions)
        print(tf.shape(sorted_predictions)[0].numpy())
        print(quantile)
        print(tf.cast(quantile,tf.float64))
        index = int(tf.shape(sorted_predictions)[0] * tf.cast(quantile,tf.float64))
        quantile_value = sorted_predictions[index]
        return quantile_value

    pt_pred_25 = calculate_quantile(pt_pred,tau_25)
    pt_pred_75 = calculate_quantile(pt_pred,tau_75)
    
    huber_loss_value = huber_loss(pt_truth, pt_pred, delta)
    quantile_loss_25 = quantile_loss(pt_truth, y_pred_25, tau_25)
    quantile_loss_75 = quantile_loss(pt_truth, y_pred_75, tau_75)
    
    complete_loss_value = huber_loss_value + quantile_loss_25 + quantile_loss_75
    complete_loss_value += 5000.*dev
    return complete_loss_value
