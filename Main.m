%-------------------------------------------------------------------------%
%  Binary Anarchich Society Algorithm (BASO) source codes                 %
%  for Feature Selection                                                  %
%                                                                         %
%  Umit Kilic                                                             %
%                                                                         %
%  email: ukilic@atu.edu.tr & umitkilic21@gmail.com                       %
%-------------------------------------------------------------------------%
warning('off');
clc;
clear;
close all;

% load the dataset
load 'datasets\car.mat';

% load the features and labels from dataset
F=features; L=labels;

%-------------------- INPUT -----------------------
% F: feature vector
% L: Labels
% N: number of population
% T: max num of iteration
%
%-------------------- OUTPUT -----------------------
% sF:       selected features
% sFNo:     Number of selected features
% sFidx:    selected features index (dataset)
% curve:    convergence curve
%
%---------------------------------------------------

N=10; T=100; 
[sF,sFNo,sFidx,curve]=bASO(F,L,N,T);

% figure();
% plot(1:T,curve);
% xlabel('Number of Iterations');
% ylabel('Fitness Value'); title('bASO'); grid on;




