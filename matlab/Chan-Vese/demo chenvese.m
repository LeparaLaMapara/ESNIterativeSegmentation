% demo for chenvese function
% Copyright (c) 2009, 
% Yue Wu @ ECE Department, Tufts University
% All Rights Reserved  
% http://sites.google.com/site/rexstribeofimageprocessing/
%%
%-- Chan & Vese method on gray and color image
%   Find contours of objects
% close all
% clear all

% I = imread('airplane_s_000003.jpg'
% Customerlized Mask
image_name  ='F:\MSC\Code\Matlab\Chan-Vese\airplane_s_000003.jpg';
path_destination ='F:\MSC\Code\Matlab\Chan-Vese\images\';
isplay(image_name)
% seg = chenvese(I,m,500,0.1,'vector'); % ability on gray image
% Built-in Mask
% seg = chenvese(image_name, path_destination,'whole+small', 200, 0.02, method); % ability on gray image
%-- End 


