import numpy as np
from ..data.transform import transform_pose_sequence
from ..data.measures import pckh, pckh_per_joint
from ..data.utils import pa16j2d


from ..utils import tensor_to_numpy

def eval_singleperson_pckh(
    model,
    fval,
    pval,
    afmat_val,
    headsize_val,
    refp=0.5,
    map_to_pa16j=None,
    pred_per_block=1,
    verbose=1,
):

    afmat_val = tensor_to_numpy(afmat_val)
    pval = tensor_to_numpy(pval)
    headsize_val = tensor_to_numpy(headsize_val)

    input_shape = fval.shape
    if len(input_shape) == 5:
        """Video clip processing."""

        num_frames = input_shape[1]
        num_batches = int(len(fval) / num_frames)

        fval = fval[0 : num_batches * num_frames]
        fval = np.reshape(fval, (num_batches, num_frames) + fval.shape[1:])

        pval = pval[0 : num_batches * num_frames]
        afmat_val = afmat_val[0 : num_batches * num_frames]
        headsize_val = headsize_val[0 : num_batches * num_frames]

    y_hats = model.forward(fval)

    num_blocks = len(y_hats) // pred_per_block

    pred = y_hats  # model.predict(inputs, batch_size=batch_size, verbose=1)

    A = afmat_val[:]
    y_true = pval[:]

    y_true = transform_pose_sequence(A.copy(), y_true, inverse=True)
    if map_to_pa16j is not None:
        y_true = y_true[:, map_to_pa16j, :]
    scores = []
    if verbose:
        print("PCKh on validation:")

    for b in range(num_blocks):

        if num_blocks > 1:
            y_pred = pred[pred_per_block * b]
        else:
            y_pred = pred

        if len(input_shape) == 5:
            """Remove the temporal dimension."""
            y_pred = y_pred[:, :, :, 0:2]
            y_pred = np.reshape(y_pred, (-1, y_pred.shape[2], y_pred.shape[3]))
        else:
            y_pred = y_pred[:, :, 0:2]

        if map_to_pa16j is not None:
            y_pred = y_pred[:, map_to_pa16j, :]
  
        y_pred = transform_pose_sequence(A.copy(), tensor_to_numpy(y_pred), inverse=True)
        s = pckh(y_true, y_pred, headsize_val, refp=refp)
        if verbose:
            print(" %.1f" % (100 * s))
        scores.append(s)

        if b == num_blocks - 1:
            if verbose:
                print("", "")
            pckh_per_joint(y_true, y_pred, headsize_val, pa16j2d, verbose=verbose)

    return scores

