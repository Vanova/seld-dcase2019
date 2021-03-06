# Contains routines for labels creation, features extraction and normalization
#


import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
# from IPython import embed
import matplotlib.pyplot as plot
import librosa
import scipy.ndimage
import math


# plot.switch_backend('agg')


class FeatureClass:
    def __init__(self, dataset_dir='', feat_label_dir='', dataset='foa', is_eval=False):
        """

        :param dataset: string, dataset name, supported: foa - ambisonic or mic- microphone format
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._feat_label_dir = feat_label_dir
        self._dataset_dir = dataset_dir
        self._dataset_combination = '{}_{}'.format(dataset, 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = 48000
        self._hop_len_s = 0.02
        self._hop_len = int(self._fs * self._hop_len_s)
        self._frame_res = self._fs / float(self._hop_len)
        self._nb_frames_1s = int(self._frame_res)

        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)

        # log-Mel features
        self._fbands = 85
        self._fmin = 0
        self._fmax = 24000

        self._dataset = dataset
        self._eps = np.spacing(np.float(1e-16))
        self._nb_channels = 4

        # Sound event classes dictionary # DCASE 2016 Task 2 sound events
        self._unique_classes = dict()
        self._unique_classes = \
            {
                'clearthroat': 2,
                'cough': 8,
                'doorslam': 9,
                'drawer': 1,
                'keyboard': 6,
                'keysDrop': 4,
                'knock': 0,
                'laughter': 10,
                'pageturn': 7,
                'phone': 3,
                'speech': 5
            }

        self._doa_resolution = 10
        self._azi_list = range(-180, 180, self._doa_resolution)
        self._length = len(self._azi_list)
        self._ele_list = range(-40, 50, self._doa_resolution)
        self._height = len(self._ele_list)

        self._audio_max_len_samples = 60 * self._fs  # TODO: Fix the audio synthesis code to always generate 60s of
        # audio. Currently it generates audio till the last active sound event, which is not always 60s long. This is a
        # quick fix to overcome that. We need this because, for processing and training we need the length of features
        # to be fixed.

        # For regression task only
        self._default_azi = 180
        self._default_ele = 50

        if self._default_azi in self._azi_list:
            print('ERROR: chosen default_azi value {} should not exist in azi_list'.format(self._default_azi))
            exit()
        if self._default_ele in self._ele_list:
            print('ERROR: chosen default_ele value {} should not exist in ele_list'.format(self._default_ele))
            exit()

        self._max_frames = int(np.ceil(self._audio_max_len_samples / float(self._hop_len)))

    def _load_audio(self, audio_path):
        # fs, audio = wav.read(audio_path)
        # audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        import soundfile
        audio, fs = soundfile.read(audio_path)
        print('duration: %.2f' % (audio.shape[0] / fs))

        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.zeros((self._audio_max_len_samples - audio.shape[0], audio.shape[1]))
            audio = np.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = np.zeros((self._max_frames, nb_bins, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(audio_input[:, ch_cnt] + self._eps, n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra[:, :, ch_cnt] = stft_ch[1:, :self._max_frames].T
        return spectra

    def _log_mel_transform(self, stft_channels):
        _nb_ch = stft_channels.shape[2]
        logmel_feat = np.zeros((self._max_frames, self._fbands, _nb_ch), dtype=float)
        for ch_cnt in range(_nb_ch):
            stft = np.abs(stft_channels[:, :, ch_cnt]) ** 2
            mel_basis = librosa.filters.mel(sr=self._fs,
                                            n_fft=self._nfft,
                                            n_mels=self._fbands,
                                            fmin=self._fmin,
                                            fmax=self._fmax)
            mel_spec = np.dot(stft, mel_basis.T[:1024, :])
            logmel_ch = librosa.logamplitude(mel_spec)
            logmel_feat[:, :, ch_cnt] = logmel_ch
        return logmel_feat

    def _extract_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(os.path.join(self._aud_dir, audio_filename))
        audio_spec = self._spectrogram(audio_in)
        # print('\t{}'.format(audio_spec.shape))
        np.save(os.path.join(self._feat_dir, '{}.npy'.format(audio_filename.split('.')[0])),
                audio_spec.reshape(self._max_frames, -1))

    def _extract_angular_phat_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(os.path.join(self._aud_dir, audio_filename))
        audio_spec = self._spectrogram(audio_in)

        cov_mat = self.empirical_cov_mat(audio_spec)

        angular_spectr = self._phi_srp_phat(cov_mat, delay=) # TODO don't know how to calculate delay

        return angular_spectr

    def _phi_srp_phat(self, ecov, delay, fbins=None):
        """Local angular spectrum function: SRP-PHAT
        Args:
            ecov  : empirical covariance matrix, indices (cctf)
            delay : the set of delays to probe, indices (dc)
            fbins : (default None) if fbins is not over all frequencies,
                    use fins to specify centers of frequency bins as discrete
                    values.
        Returns:
            phi   : local angular spectrum function, indices (dt),
                    here 'd' is the index of delay
        """
        # compute inverse of empirical covariance matrix
        nch, _, nframe, nfbin = ecov.shape

        mask = np.asarray([[c < d for d in xrange(nch)] for c in xrange(nch)])
        ecov_upper_tri = ecov[mask]
        cpsd_phat = np.zeros(ecov_upper_tri.shape, dtype=ecov_upper_tri.dtype)
        ampl = np.abs(ecov_upper_tri)
        non_zero_mask = ampl > 1e-20
        cpsd_phat[non_zero_mask] = ecov_upper_tri[non_zero_mask] / ampl[non_zero_mask]

        phi = np.zeros((len(delay), nframe))
        for i in xrange(len(delay)):
            if fbins is None:
                stv = self.steering_vector(delay[i], nfbin)
            else:
                stv = self.steering_vector(delay[i], fbins=fbins)

            x = np.asarray([stv[c] * stv[d].conj() for c in xrange(nch)
                            for d in xrange(nch)
                            if c < d])
            phi[i] = np.einsum('itf,if->t', cpsd_phat, x.conj(),
                               optimize='optimal').real / nfbin / len(x)
        return phi

    def steering_vector(self, delay, win_size=0, fbins=None, fs=None):
        """Compute the steering vector.
        One and only one of the conditions are true:
            - win_size != 0
            - fbins is not None
        Args:
            delay : delay of each channel (see compute_delay),
                    unit is second if fs is not None, otherwise sample
            win_size : (default 0) window (FFT) size. If zero, use fbins.
            fbins : (default None) center of frequency bins, as discrete value.
            fs    : (default None) sample rate
        Returns:
            stv   : steering vector, indices (cf)
        """
        assert (win_size != 0) != (fbins is not None)
        delay = np.asarray(delay)
        if fs is not None:
            delay *= fs  # to discrete-time value
        if fbins is None:
            fbins = np.fft.fftfreq(win_size)
        return np.exp(-2j * math.pi * np.outer(delay, fbins))

    def empirical_cov_mat(self, tf, tw=2, fw=2):
        """Empirical covariance matrix
        Args:
            tf  : multi-channel time-frequency domain signal, indices (ctf)
            tw  : (default 2) half width of neighbor area in time domain,
                  including center
            fw  : (default 2) half width of neighbor area in freq domain,
                  including center
        Returns:
            ecov: empirical covariance matrix, indices (cctf)
        """
        _apply_conv = scipy.ndimage.filters.convolve

        tf = tf.transpose(2, 0, 1)

        # covariance matrix without windowing
        cov = np.einsum('ctf,dtf->cdtf', tf, tf.conj())

        # apply windowing by convolution
        # compute convolution window
        kernel = np.einsum('t,f->tf', np.hanning(tw * 2 + 1)[1:-1],
                           np.hanning(fw * 2 + 1)[1:-1])
        kernel = kernel / np.sum(kernel)  # normalize

        # apply to each channel pair
        ecov = np.zeros(cov.shape, dtype=cov.dtype)
        for i in xrange(len(tf)):
            for j in xrange(len(tf)):
                rpart = _apply_conv(cov[i, j, :, :].real, kernel, mode='nearest')
                ipart = _apply_conv(cov[i, j, :, :].imag, kernel, mode='nearest')
                ecov[i, j, :, :] = rpart + 1j * ipart
        return ecov

    # OUTPUT LABELS
    def read_desc_file(self, desc_filename, in_sec=False):
        desc_file = {
            'class': list(), 'start': list(), 'end': list(), 'ele': list(), 'azi': list()
        }
        fid = open(desc_filename, 'r')
        next(fid)
        for line in fid:
            split_line = line.strip().split(',')
            desc_file['class'].append(split_line[0])
            # desc_file['class'].append(split_line[0].split('.')[0][:-3])
            if in_sec:
                # return onset-offset time in seconds
                desc_file['start'].append(float(split_line[1]))
                desc_file['end'].append(float(split_line[2]))
            else:
                # return onset-offset time in frames
                desc_file['start'].append(int(np.floor(float(split_line[1]) * self._frame_res)))
                desc_file['end'].append(int(np.ceil(float(split_line[2]) * self._frame_res)))
            desc_file['ele'].append(int(split_line[3]))
            desc_file['azi'].append(int(split_line[4]))
        fid.close()
        return desc_file

    def get_list_index(self, azi, ele):
        azi = (azi - self._azi_list[0]) // 10
        ele = (ele - self._ele_list[0]) // 10
        return azi * self._height + ele

    def get_matrix_index(self, ind):
        azi, ele = ind // self._height, ind % self._height
        azi = (azi * 10 + self._azi_list[0])
        ele = (ele * 10 + self._ele_list[0])
        return azi, ele

    def _get_doa_labels_regr(self, _desc_file):
        azi_label = self._default_azi * np.ones((self._max_frames, len(self._unique_classes)))
        ele_label = self._default_ele * np.ones((self._max_frames, len(self._unique_classes)))
        for i, ele_ang in enumerate(_desc_file['ele']):
            start_frame = _desc_file['start'][i]
            end_frame = self._max_frames if _desc_file['end'][i] > self._max_frames else _desc_file['end'][i]
            azi_ang = _desc_file['azi'][i]
            class_ind = self._unique_classes[_desc_file['class'][i]]
            if (azi_ang >= self._azi_list[0]) & (azi_ang <= self._azi_list[-1]) & \
                    (ele_ang >= self._ele_list[0]) & (ele_ang <= self._ele_list[-1]):
                azi_label[start_frame:end_frame + 1, class_ind] = azi_ang
                ele_label[start_frame:end_frame + 1, class_ind] = ele_ang
            else:
                print('bad_angle {} {}'.format(azi_ang, ele_ang))
        doa_label_regr = np.concatenate((azi_label, ele_label), axis=1)
        return doa_label_regr

    def _get_se_labels(self, _desc_file):
        se_label = np.zeros((self._max_frames, len(self._unique_classes)))
        for i, se_class in enumerate(_desc_file['class']):
            start_frame = _desc_file['start'][i]
            end_frame = self._max_frames if _desc_file['end'][i] > self._max_frames else _desc_file['end'][i]
            se_label[start_frame:end_frame + 1, self._unique_classes[se_class]] = 1
        return se_label

    def get_labels_for_file(self, _desc_file):
        """
        Reads description csv file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: csv file
        :return: label_mat: labels of the format [sed_label, doa_label],
        where sed_label is of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
        where doa_labels is of dimension [nb_frames, 2*nb_classes], nb_classes each for azimuth and elevation angles,
        if active, the DOA values will be in degrees, else, it will contain default doa values given by
        self._default_ele and self._default_azi
        """

        se_label = self._get_se_labels(_desc_file)
        doa_label = self._get_doa_labels_regr(_desc_file)
        label_mat = np.concatenate((se_label, doa_label), axis=1)
        # print(label_mat.shape)
        return label_mat

    def get_clas_labels_for_file(self, _desc_file):
        """
        Reads description file and returns classification format labels for SELD

        :param _desc_file: csv file
        :return: _labels: matrix of SELD labels of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                          which is 1 for active sound event and location else zero
        """

        _labels = np.zeros((self._max_frames, len(self._unique_classes), len(self._azi_list) * len(self._ele_list)))
        for _ind, _start_frame in enumerate(_desc_file['start']):
            _tmp_class = self._unique_classes[_desc_file['class'][_ind]]
            _tmp_azi = _desc_file['azi'][_ind]
            _tmp_ele = _desc_file['ele'][_ind]
            _tmp_end = self._max_frames if _desc_file['end'][_ind] > self._max_frames else _desc_file['end'][_ind]
            _tmp_ind = self.get_list_index(_tmp_azi, _tmp_ele)
            _labels[_start_frame:_tmp_end + 1, _tmp_class, _tmp_ind] = 1

        return _labels

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def extract_all_feature(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)

        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))

        for file_cnt, file_name in enumerate(os.listdir(self._aud_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            # self._extract_spectrogram_for_file(wav_filename)
            # TODO check angular TDOA
            self._extract_angular_phat_for_file(wav_filename)

    def preprocess_features(self):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()

        # pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))

        else:

            print('Estimating normalization weights on Log-Mel and DOA phase features:')
            print('\t\tfeat_dir: {}'.format(self._feat_dir))

            spec_scaler = preprocessing.StandardScaler()
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print('{}: {}'.format(file_cnt, file_name))
                feat_file = np.load(os.path.join(self._feat_dir, file_name))

                # SED features: Log-Mel
                feat_chs = feat_file.reshape(-1, self._nfft // 2, self._nb_channels)
                logmel_feat = self._log_mel_transform(feat_chs)
                logmel_feat = logmel_feat.reshape(self._max_frames, -1)
                # DOA features are phases
                phase = np.angle(feat_file)

                spec_scaler.partial_fit(np.concatenate((logmel_feat, phase), axis=1))
                del feat_file
            joblib.dump(spec_scaler, normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._feat_dir, file_name))

            # SED features: Log-Mel
            feat_chs = feat_file.reshape(-1, self._nfft // 2, self._nb_channels)
            logmel_feat = self._log_mel_transform(feat_chs)
            logmel_feat = logmel_feat.reshape(self._max_frames, -1)
            # DOA features are phases
            phase = np.angle(feat_file)

            feat_file = spec_scaler.transform(np.concatenate((logmel_feat, phase), axis=1))
            np.save(
                os.path.join(self._feat_dir_norm, file_name),
                feat_file
            )
            del feat_file

        print('normalized files written to {}'.format(self._feat_dir_norm))

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            desc_file = self.read_desc_file(os.path.join(self._desc_dir, file_name))
            label_mat = self.get_labels_for_file(desc_file)
            np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # ------------------------------- Misc public functions -------------------------------
    def get_classes(self):
        return self._unique_classes

    def get_normalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format(self._dataset_combination)
        )

    def get_unnormalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}'.format(self._dataset_combination)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir, '{}_label'.format(self._dataset_combination)
            )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_svd_doa_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_svd_doa'.format(self._dataset)
        )

    def get_default_azi_ele_regr(self):
        return self._default_azi, self._default_ele

    def get_nb_channels(self):
        return self._nb_channels

    def nb_frames_1s(self):
        return self._nb_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_azi_ele_list(self):
        return self._azi_list, self._ele_list

    def get_nb_frames(self):
        return self._max_frames


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)
