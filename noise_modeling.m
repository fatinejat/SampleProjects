clear all;
close all;

% frequencies = 1: 1: 100;
% amplitudes = 20*gaussmf(frequencies, [2, 20]) ;

frequencies = [10, 17, 38, 85];
amplitudes = [40, 80, 10, 66];

sf = 1000;
duration = 5;

t = linspace(0, duration, duration*sf);


noise_amplitude = 100;
inv_alpha = 1.8;
cn = dsp.ColoredNoise('InverseFrequencyPower', inv_alpha, 'SamplesPerFrame', duration*sf);
noise = noise_amplitude*cn()';

signal = zeros(1, length(t));
for i = 1: length(frequencies)
    signal = signal + amplitudes(i)*sin(2*pi*t*frequencies(i));
end

noisy_signal = signal + noise;

m = ar(noisy_signal, 1);
ar_denoised_signal = noisy_signal + m.A(2)*[noisy_signal(2: end), 0];
loess_denoised_signal = smooth(noisy_signal, 0.5, 'rlowess')';

figure; 
subplot(1, 3, 1); hold on;
plot(t, signal);
plot(t, noisy_signal);
plot(t, ar_denoised_signal);
plot(t, loess_denoised_signal);

title('Signals');
set(findall(gca, 'Type', 'Line'),'LineWidth',2);

legend({'Signal', ['Noise Added \alpha: ', num2str(inv_alpha)], 'AR Denoised', 'R Lowess Denoised'});



subplot(1, 3, 2); hold on;
[pxx, f] = periodogram(signal, [], [], sf);
plot(log(f), log(pxx));

[pxx, f] = periodogram(noisy_signal, [], [], sf);
plot(log(f), log(pxx));

[pxx, f] = periodogram(ar_denoised_signal, [], [], sf);
plot(log(f), log(pxx));

[pxx, f] = periodogram(loess_denoised_signal, [], [], sf);
plot(log(f), log(pxx));

% xlim([0, max(frequencies)*2]);

title('PSD');
set(findall(gca, 'Type', 'Line'),'LineWidth',2);


subplot(1, 3, 3); hold on;
plot(autocorr(signal, 100));
plot(autocorr(noisy_signal, 100));
plot(autocorr(ar_denoised_signal, 100));
plot(autocorr(loess_denoised_signal, 100));

title('Auto-Correlation');

set(findall(gca, 'Type', 'Line'),'LineWidth',2);