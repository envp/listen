import numpy as np

from scipy.signal import butter, lfilter, hamming

class Filter(object):
    @staticmethod
    def hz2mel(hz):
        """Convert a value in Hertz to Mels
        :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 1125 * np.log10(1 + hz / 700.0)

    @staticmethod
    def mel2hz(mel):
        """Convert a value in Mels to Hertz
        :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700 * (10 ** (mel / 1125.0) - 1)

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @classmethod
    def butter_bandpass_filter(cls, data, lowcut, highcut, fs, order=5):
        b, a = cls.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    @classmethod
    def get_filterbanks(cls, nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond to fft bin_numbers.
        The filters are returned as an array of size nfilt * (nfft/2 + 1)

        :param nfilt: the number of filters in the filterbank, default 20.

        :param nfft: the FFT size. Default is 512.

        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.

        :param lowfreq: lowest band edge of mel filters, default 0 Hz

        :param highfreq: highest band edge of mel filters, default samplerate/2

        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or samplerate / 2
        assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        lowmel = Filter.hz2mel(lowfreq)
        highmel = Filter.hz2mel(highfreq)
        melpoints = np.linspace(lowmel, highmel, nfilt + 2)
        # our points are in Hz, but we use fft bin_numbers, so we have to convert
        #  from Hz to fft bin_number number
        bin_number = np.floor((nfft + 1) * Filter.mel2hz(melpoints) / samplerate)

        fbank = np.zeros([nfilt, nfft // 2])
        for j in range(0, nfilt):
            for i in range(int(bin_number[j]), int(bin_number[j + 1])):
                fbank[j, i] = (i - bin_number[j]) / \
                    (bin_number[j + 1] - bin_number[j])
            for i in range(int(bin_number[j + 1]), int(bin_number[j + 2])):
                fbank[j, i] = (bin_number[j + 2] - i) / \
                    (bin_number[j + 2] - bin_number[j + 1])
        return fbank

    @classmethod
    def create_mel_filter(cls, fft_size, n_freq_components=64, start_freq=300, end_freq=8000, samplerate=44100):
        """Creates a filter to convolve with the spectrogram to get out mels
        """
        mel_inversion_filter = cls.get_filterbanks(nfilt=n_freq_components,
                                                   nfft=fft_size, samplerate=samplerate,
                                                   lowfreq=start_freq, highfreq=end_freq)
        # Normalize filter
        mel_filter = mel_inversion_filter.T / mel_inversion_filter.sum(axis=1)

        return mel_filter, mel_inversion_filter

    @staticmethod
    def time_stretch(x, ts_ratio, L=1024, H=256, win=hamming):
        syn_hop = int(H * ts_ratio)
        N = len(x)
        w = win(L)
        gain = 1. / (L * np.sum((win(L) * win(L))) / syn_hop)

        unwrapdata = 2 * np.pi * H / L * np.arange(0, L).T

        yangle, ysangle = np.zeros(L), np.zeros(L)

        ys = np.zeros(L, dtype=np.complex)
        yprevwin = np.zeros(L - syn_hop, dtype=np.complex)

        first_time = True
        y = np.array(0)

        for i in np.arange(0, N - L, H):

            yprevangle = yangle

            yfft = np.fft.fft(w * x[i : i + L])
            ymag, yangle = np.abs(yfft), np.angle(yfft)

            yunwrap = (yangle - yprevangle) - unwrapdata
            yunwrap = yunwrap - np.round(yunwrap / (2.*np.pi)) * 2 * np.pi
            yunwrap = (yunwrap + unwrapdata) * ts_ratio

            if first_time:
                ysangle = yangle
                first_time = False
            else:	ysangle += yunwrap

            ys.real, ys.imag = np.cos(ysangle), np.sin(ysangle)
            ys *= ymag
            ywin = np.fft.ifft(w * ys)

            overlap_add = np.hstack((ywin[:L - syn_hop] + yprevwin,\
                        ywin[L - syn_hop: ]))
            yistfft = overlap_add[: syn_hop]
            yprevwin = overlap_add[syn_hop : ]

            yistfft = yistfft * gain
            y = np.hstack((y, yistfft))

        return np.real(y * np.max(np.abs(x)) / np.max(np.abs(y)))
