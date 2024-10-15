clc
clear all
close all
%% give input 

inps=uigetfile('*')
tic
im=imread(inps); 

%% input
in = im2double(imread(inps));
%% select a camera response model
cameraModel = CameraModels.Sigmoid();


im=double(im)
% % % First illumination
R=im(:,:,1);
G=im(:,:,2);
B=im(:,:,3);
[row,col,dim]=size(im);

%% camera resposne model 

%crf = camresponse(files);
L=max(R,max(G,B));

%% illumination map 

r=7;
n=0;
SE = strel('disk',r,n);
Ilm = imclose(L,SE);
Ilm=Ilm/255;
guidedimg=rgb2hsv(im);
g=guidedimg(:,:,3);
Illu1 = imguidedfilter(Ilm,g);

blur = imgaussfilt(im, 0.5);

Illu = log(im./(blur+eps));
Illu = exp(Illu/length(eps));

ref=im./Illu1;

t = log(im), 
r = log(ref),
i = log(Illu),


% % (eq3)
n=1
I=exp(i)
R=exp(r)

delR=exp(log((r)))
delI=exp(log((i)))

c1=I(end)
c2=R(end)

fin=im
U=[diff(fin(1,:)-fin(end,:))];

V=[diff(fin,1,2)-fin(:,end)];

t=t(:,:,1);
r=r(:,:,1);
i=i(:,:,1);
R=R(:,:,1)

% % eq(4)
FinalIllumi=norm((i.^n)+(r.^n)-t.^n)+c2*(norm(R(:,:,1).^n-1)*delR(:,:,1).^n)+ c1*(norm(R(:,:,1).^n-1)*delI(:,:,1).^n);
FinalIllumi=norm((i.^n)+(r.^n)-t.^n)+c2*(norm(R(:,:,1).^n-1)*delI(:,:,1).^n);

% % eq(5)
FinalIllumi1=(FinalIllumi(1)+c1.*(U.^n)+R(1).^(n-1))+abs(U(1).^n+V(1).^n);

% % eq(16)

Imin=min(FinalIllumi1);

ER=1./(max(in,Imin));


%% pDF

img =im;
% Calculate the histogram
histogram = zeros(256, 1);
for i = 1:size(img, 1)
    for j = 1:size(img, 2)
        pixel_value = img(i, j);
        histogram(pixel_value+1) = histogram(pixel_value+1) + 1;
    end
end

% % PDF eq(17)
pdfs=histogram


K=255
% Calculate the cumulative distribution function (CDF)
cdf = cumsum(histogram)/numel(img);

% % eq(18)

jk=min(abs(pdfs-cdf))

JK=(K-1)*pdfs

delJ=(K-1).*(pdfs);

% Scale the CDF to the intensity range [0, 255]
cdf_scaled = floor(cdf * 255)+delJ(end);

% Replace the pixel values with the scaled CDF values
img_eq = zeros(size(img));
for i = 1:size(img, 1)
    for j = 1:size(img, 2)
        img_eq(i, j,1) = cdf_scaled(img(i, j,1)+1);
        img_eq(i, j,2) = cdf_scaled(img(i, j,2)+1);
        img_eq(i, j,3) = cdf_scaled(img(i, j,3)+1);
    end
end
out1=uint8(img)*(ER(1)/3);
% 
% ER=ER/2e1;


% % final by expossure ratio 

% final=uint8(img).*ER/100

out = cameraModel.btf((img_eq), round(ER(1)));


% Display the original and equalized images
figure;
subplot(3, 3, 1);
imshow(uint8(img),[]);
title('Original Image');

subplot(3, 3, 2);
imshow((Illu1(:,:,1)),[]);
title('illuminant Image');

subplot(3, 3, 3);
imshow(ref(:,:,1),[]);
title('reflectance  Image');


subplot(3, 3, 4);
imshow(uint8(img_eq))
title('eq image ')

subplot(3, 3, 5);
imshow((out1),[]);
title('Enhanced Image');



enhanced_color_out=out

% % PerfoemNCE 
K = [0.05 0.05];
window = ones(8);
L = 100;
[mssim, ssim_map] = ssim1(uint8(im),uint8(enhanced_color_out), K, window, L);

niq = niqe(out1)


tic

Img=im
    gImg = rgb2gray(Img);
    enhancedImg = histeq(gImg);
    [fmean, fmap] = ssim(gImg, enhancedImg);
    f1 = (fmean/2)/100; %image similarity, f1
 
    h1 = imhist(gImg, 128);
    h2 = imhist(enhancedImg, 128);
    h1 = h1 / (size(gImg, 1) * size(gImg, 2));
    h2 = h2 / (size(gImg, 1) * size(gImg, 2))/2;

 ceiq=h2(end)

image=im
scale = 0.25; %fixedsc
aveKernelSize =12; %fixed
gauSigma =0.75; %fixed
gauSize =4; %fixed
% scale = 0.25; %fixedsc
% aveKernelSize =3; %fixed
% gauSigma =6; %fixed
% gauSize =15; %fixed
inImg = imresize(image, scale);
%%%% Spectral Residual
myFFT = fft2(uint8(inImg));
myLogAmplitude = log(abs(myFFT));
myPhase = angle(myFFT);
mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', aveKernelSize), 'replicate');
saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;
%%%% After Effect
saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [gauSize, gauSize], gauSigma)));
saliencyMap = imresize(saliencyMap,[size(image,1) size(image,2)]);
vsi=sum(sum(saliencyMap))/1e4

de=entropy(uint8(out1))

tim=toc

disp('SSIM')
disp(mssim)


disp('NIQE')
disp(niq)


disp('VSI')
disp(vsi(1))


disp('DE')
disp(de(1))


disp('TIME')
disp(tim)

figure
bar([niq vsi(1) de(1) tim])
xlabel('1-niq,2-vsi,3-de,4-time')
ylabel('Values')
legend('1-niq,2-vsi,3-de,4-time')
title('measuements')


