#
# A wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
from metrics import evaluation_metrics
import keras_model
from keras.models import load_model
import parameter
import time

plot.switch_backend('agg')


def collect_test_labels(_data_gen_test, _data_out, quick_test):
    # Collecting ground truth for test data
    nb_batch = 2 if quick_test else _data_gen_test.get_total_batches_in_data()

    batch_size = _data_out[0][0]
    gt_sed = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2]))
    gt_doa = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2]))

    print("nb_batch in test: {}".format(nb_batch))
    cnt = 0
    for tmp_feat, tmp_label in _data_gen_test.generate():
        gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
        gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[1]
        cnt = cnt + 1
        if cnt == nb_batch:
            break
    return gt_sed.astype(int), gt_doa


def main(argv):
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameter.get_params(task_id)

    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        # test_splits = [1, 2, 3, 4]
        # val_splits = [2, 3, 4, 1]
        # train_splits = [[3, 4], [4, 1], [1, 2], [2, 3]]
        # TODO for debug only
        test_splits = [1]
        val_splits = [1]
        train_splits = [[1, 1]]

        # SUGGESTION: Considering the long training time, major tuning of the method can be done on the first split.
        # Once you finlaize the method you can evaluate its performance on the complete cross-validation splits
        # test_splits = [1]
        # val_splits = [2]
        # train_splits = [[3, 4]]

    elif params['mode'] == 'eval':
        test_splits = [0]
        val_splits = [1]
        train_splits = [[2, 3, 4]]

    # ------------------  Calculate metric scores for unseen test split ---------------------------------
    print('Loading testing dataset:')
    data_gen_test = cls_data_generator.DataGenerator(
        dataset=params['dataset'], split=split, batch_size=params['batch_size'], seq_len=params['sequence_length'],
        feat_label_dir=params['feat_label_dir'], shuffle=False, per_file=params['dcase_output'],
        is_eval=True if params['mode'] is 'eval' else False
    )

    # print('\nLoading the best model and predicting results on the testing split')
    # model = load_model('{}_model.h5'.format(unique_name))
    # pred_test = model.predict_generator(
    #     generator=data_gen_test.generate(),
    #     steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
    #     verbose=2
    # )

    test_sed_pred = evaluation_metrics.reshape_3Dto2D(pred_test[0]) > 0.5
    test_doa_pred = evaluation_metrics.reshape_3Dto2D(pred_test[1])

    # rescaling the elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
    test_doa_pred[:, nb_classes:] = test_doa_pred[:, nb_classes:] / (180. / def_elevation)

    if params['dcase_output']:
        # Dump results in DCASE output format for calculating final scores
        dcase_dump_folder = os.path.join(params['dcase_dir'],
                                         '{}_{}_{}'.format(task_id, params['dataset'], params['mode']))
        cls_feature_class.create_folder(dcase_dump_folder)
        print('Dumping recording-wise results in: {}'.format(dcase_dump_folder))

        test_filelist = data_gen_test.get_filelist()
        # Number of frames for a 60 second audio with 20ms hop length = 3000 frames
        max_frames_with_content = data_gen_test.get_nb_frames()

        # Number of frames in one batch (batch_size* sequence_length) consists of all the 3000 frames above with
        # zero padding in the remaining frames
        frames_per_file = data_gen_test.get_frame_per_file()

        for file_cnt in range(test_sed_pred.shape[0] // frames_per_file):
            output_file = os.path.join(dcase_dump_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            dc = file_cnt * frames_per_file
            output_dict = evaluation_metrics.regression_label_format_to_output_format(
                data_gen_test,
                test_sed_pred[dc:dc + max_frames_with_content, :],
                test_doa_pred[dc:dc + max_frames_with_content, :] * 180 / np.pi
            )
            evaluation_metrics.write_output_format_file(output_file, output_dict)

    if params['mode'] is 'dev':
        _, _, test_data_out = data_gen_test.get_data_sizes()
        test_gt = collect_test_labels(data_gen_test, test_data_out, params['quick_test'])
        test_sed_gt = evaluation_metrics.reshape_3Dto2D(test_gt[0])
        test_doa_gt = evaluation_metrics.reshape_3Dto2D(test_gt[1])
        # rescaling the reference elevation from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        test_doa_gt[:, nb_classes:] = test_doa_gt[:, nb_classes:] / (180. / def_elevation)

        test_sed_loss = evaluation_metrics.compute_sed_scores(test_sed_pred, test_sed_gt,
                                                              data_gen_test.nb_frames_1s())
        test_doa_loss = evaluation_metrics.compute_doa_scores_regr(test_doa_pred, test_doa_gt, test_sed_pred,
                                                                   test_sed_gt)
        test_metric_loss = evaluation_metrics.compute_seld_metric(test_sed_loss, test_doa_loss)

        avg_scores_test.append(
            [test_sed_loss[0], test_sed_loss[1], test_doa_loss[0], test_doa_loss[1], test_metric_loss])
        print('Results on test split:')
        print('\tSELD_score: {},  '.format(test_metric_loss))
        print('\tDOA Metrics: DOA_error: {}, frame_recall: {}'.format(test_doa_loss[0], test_doa_loss[1]))
        print('\tSED Metrics: ER_overall: {}, F1_overall: {}\n'.format(test_sed_loss[0], test_sed_loss[1]))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
