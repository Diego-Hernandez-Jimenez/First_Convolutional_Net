

% ********************************************************************
%  Author ("base" script): Luis Jáñez Escalada   
% (C) 2018-21  Luis Jáñez Escalada 
% Author (edited script): Diego Hernández Jiménez
% ********************************************************************



%%%%%%%%% Start script
global plotObj
clearvars
%clc

% To measure execution time
datetime
tic
timerVal_60 = tic;
 
% Just to make sure nothing affects the current program
close all
close all hidden

fopen('all');
fclose ('all');


 

%**************************************************************************
%**************************************************************************
%
%      1. LOAD AND PREPARE DATA SET
%
%**************************************************************************
%**************************************************************************



digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
        'nndatasets','DigitDataset');
 
digitData = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    
 

% Show some images
 
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(digitData.Files{perm(i)});
end


% Check number of images in each category

CountLabel = digitData.countEachLabel;

% CountLabel =
%   10×2 table
%     Label    Count
%     _____    _____
%     0        1000 
%     1        1000 
%     2        1000 
%     3        1000 
%     4        1000 
%     5        1000 
%     6        1000 
%     7        1000 
%     8        1000 
%     9        1000 



% Check image size

img = readimage(digitData,1);
size(img)

% size(img)
% ans =
%     28    28


%**************************************************************************
%  SPLIT TRAINING AND TEST SET I
%**************************************************************************

% Divide the data into training and test sets, so that each category in 
% the training set has 750 images and the test set has the remaining images 
% from each label.

trainingNumFiles = 750;
rng(1) % For reproducibility
% splitEachLabel splits the image files in digitData into two new datastores, 
% trainDigitData and testDigitData.
[trainDigitData,testDigitData] = ...
    splitEachLabel( digitData, trainingNumFiles,'randomize' )  ;

%%

%**************************************************************************
%**************************************************************************
%
%                    2. DEFINE LAYERS
%
%**************************************************************************
%**************************************************************************



%  ------
% Layer 1
%  ------


% I1_layer
I1_nrows = 28; 
I1_ncols = 28; 
I1_nchan = 1;   
I1_nn = I1_nrows*I1_ncols*I1_nchan; % number of outputs or neurons
I1_layer = imageInputLayer([I1_nrows I1_ncols I1_nchan],...
  'Name','I1_layer',...
  'Normalization','none');

%% 


%  ------
%  layer 2
%  ------

% C2
C2_fn = 20; % feature maps / kernels
C2_ks = [5 5] ; % kernel size
C2_layer = convolution2dLayer(C2_ks, C2_fn,...
    'Name','C2_layer',...
    'Stride',[1 1],...
    'Padding' ,[0 0 0 0],...
    'NumChannels','auto',...
    'BiasLearnRateFactor', 1,...
    'WeightL2Factor',1,...
    'BiasL2Factor', 1) ;


% C2 initialization
    
% initial standard deviation for weights and biases
C2_SDinitWeights=0.01 ;

% Manually initialize the weights from a Gaussian distribution with standard deviation of 0.0001.
C2_layer.Weights =  randn([C2_ks(1) C2_ks(2) I1_nchan C2_fn]) * C2_SDinitWeights ;



% show initial weights and biases in convolutional layer

w1 = C2_layer.Weights;
% Scale and resize the weights for visualization
w1 = mat2gray(w1);
% amplia la imagen
w1x4(:,:,1,:) = imresize(w1(:,:,1,:),4,'Method','nearest');
% w1(1,:,3,2)   (columna, fila, profundidad, nº de feature map)
figure
montage(w1x4,'BorderSize',[1 1],'BackgroundColor','yellow' )
title('First convolutional layer weights (pre-training) zoom x 4')

%% 


%  ------
%  layer 3
%  ------

% R3 RELU layer
R3_layer = reluLayer('Name','C3_ReLU');

%%

%  ------
%  layerr 4
%  ------

% P4 MaxPooling   
P4_ks = [2 2]; % kernel size
P4_stride = [2 2];
P4_pad = [0 0];  %=ceil((S4_is - P4_ks + 1) ./ P4_ss);
P4_layer = maxPooling2dLayer(P4_ks,...
    'Name','P4_maxpool',...
    'Stride',P4_stride,...
    'Padding',P4_pad);

%%

%  ------
%  layer 5
%  ------

F5_s = 10; % nº neurons
F5_layer = fullyConnectedLayer(F5_s, 'Name','F5_FC') ;

%%


%  ------
%  layer 6
%  ------

%S6 SoftMax layer
S6_layer = softmaxLayer('Name','S6_layer');

%%

%  ------
%  layer 7
%  ------

%S7 classificationLayer()
C7_layer= classificationLayer('Name','C7_layer');

%%



%-------------------------------------------------------
% CREATES THE NETWORK BY ASSEMBLING THE NEWLY DEFINED LAYERS 
%-------------------------------------------------------

Janet = [  I1_layer
           C2_layer
           R3_layer
           P4_layer
           F5_layer
           S6_layer
           C7_layer    ]



analyzeNetwork(Janet) % graphic output


%%

%**************************************************************************
%**************************************************************************
%
%          3.  TRAINING OPTIONS
%
%**************************************************************************
%**************************************************************************

% Auxiliary graphic task: 
% Functions to paint the learning process; 
% are at the end of this program
functions = { ...
    @plotTrainingAccuracy, ...
    @(info) stopTrainingAtThreshold(info,98), ...
    @(info) stopIfAccuracyNotImproving(info,10)};



% Specify the Training Options
% 'sgdm' :stochastic gradient descent with momentum
options = trainingOptions('sgdm',...
    'MaxEpochs',30, ...
    'ValidationData',testDigitData, ...
    'ValidationFrequency',25, ...
    'MiniBatchSize',50,...
	'InitialLearnRate',0.0001,...
    'Momentum', 0.7, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','cpu',...
    'OutputFcn',functions);


%%

%**************************************************************************
%**************************************************************************
%
%     4.  TRAINING
%
%**************************************************************************
%**************************************************************************

figure
axis([0 4500 0 100])
[convnet,traininfo]= trainNetwork(trainDigitData,Janet,options);


% traininfo
% Information on the training, returned as a structure with the following fields.
%     TrainingLoss — Loss function value at each iteration
%     TrainingAccuracy — Training accuracy at each iteration if network is a classification network
%     TrainingRMSE — Training RMSE at each iteration if network is a regression network
%     BaseLearnRate — The learning rate at each iteration

%%

%**************************************************************************

% Check the convolution kernels learned in the 1st convolution layer.
 
convnet.Layers(2)

w1 = convnet.Layers(2).Weights;


% Scale and resize the weights for visualization
w1 = mat2gray(w1);
[s1,s2,s3,s4]= size (w1) ;


w1x4(:,:,1,:) = imresize(w1(:,:,1,:),4,'Method','nearest');

% Display WEIGHTS of 2nd feature map  at the first layer  of network weights. 
% There are 96 individual sets of weights in the first layer.
% w1(1,:,3,2)   (columna, fila, profundidad, nº de feature map)
figure
montage(w1x4,'BorderSize',[1 1],'BackgroundColor','yellow' )
title('First convolutional layer weights (post-training) zoom x 4')


%%

%**************************************************************************
%**************************************************************************
%
%     5.  CLASIFICAR LAS IMÁGENES DE PRUEBA Y CALCULAR LA PRECISIÓN
%
%**************************************************************************
%**************************************************************************


% Run the trained network on the test set that was not used to train 
% the network and predict the image labels (digits).
% datos propios de test    
MyDigitData = imageDatastore('C:\Users\Diego\Desktop\matprojects\MyDigitDatasetCopy', ...
        'IncludeSubfolders',true,'LabelSource','foldernames');

% Clasificar las imágenes de test
YTest = classify(convnet,MyDigitData);

TTest = MyDigitData.Labels;

% Calcular exactitud en datos de TEST.

accuracytest = mean(YTest == TTest);
fprintf (1,' \r\n %-s %f8 \r\n', 'TEST SET ACCURACY: ',accuracytest)  ;


fprintf (1,' \r\n %-s %f8 \r\n', 'CONFUSION MATRIX ' )  ;
confMat = confusionmat(TTest,YTest);

% to see it easily
confusionchart(TTest,YTest)
title('confusion matrix');
[TTest YTest]


%%

% SHOW ACTIVATION OF FIRST DIGITS (see exercise 5)

firstnums = 1:5:50;
% firstnums = 1:1000:10000; % if you want to show MNIST digits

for layer = 1:7
    all_activs = [];
    for dig = 1:10
        % filename = digitData.Files{firstnums(dig)}; % if you want to show
        % MNIST digits
        filename = MyDigitData.Files{firstnums(dig)};
        im = imread(filename);
        layername = convnet.Layers(layer).Name;
        activ = activations(convnet,im,layername);
        if layer == 1
            firstmatrix = activ(:,:,1);
        else % conditional allows to extract activations from matrices or neurons ~= than first
            firstmatrix = activ(:,:,4);
        end
        all_activs = cat(3,all_activs,firstmatrix); % concatenate activations in 3d array
    end
    figure
    montage(all_activs,'Size',[2 5],'BorderSize',[1 1],'BackgroundColor','yellow' )
    title(strcat(layername,' activations for the first ten different digits'),...
        'Interpreter','none')
end

%%
%**************************************************************************
%  EXECUTION TIME
%**************************************************************************

fopen('all');
fclose ('all');


datetime
toc 
toc(timerVal_60)  % displays the time elapsed since the tic command corresponding to timerVal.
elapsedTime = toc(timerVal_60);  % returns the elapsed time since the tic command corresponding to timerVal.
%elapsedTime = toc ;% returns the elapsed time in a variable.
minuteselapsed=elapsedTime/60;
fprintf (1,'\r\n %-s %0.2f ','Minutos de ejecución: ', minuteselapsed );
fprintf (1,'\r\n %-s %0.2f \r\n','Horas de ejecución: ', minuteselapsed/60 );

%*************************************************************************
%  END
%*************************************************************************




function plotTrainingAccuracy(info)
% info
persistent plotObj

if info.State == "start"
    plotObj = animatedline;
 

    xlabel("Iteration")
    ylabel("Training Accuracy")
elseif info.State == "iteration"
    addpoints(plotObj,info.Iteration,info.TrainingAccuracy)
    drawnow limitrate nocallbacks
else
    fprintf (1,'\r\n %-s %i\r\n','FINALIZÓ EL APRENDIZAJE ');
end

end

function stop = stopTrainingAtThreshold(info,thr)

stop = false;
if info.State ~= "iteration"
    return
end

persistent iterationAccuracy

% Append accuracy for this iteration
iterationAccuracy = [iterationAccuracy info.TrainingAccuracy];

% Evaluate mean of iteration accuracy and remove oldest entry
if numel(iterationAccuracy) == 50
    stop = mean(iterationAccuracy) > thr;

    iterationAccuracy(1) = [];
end

end


function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent niters

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    niters = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if round(info.ValidationAccuracy,1) > round(bestValAccuracy,1)
        niters = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        niters = niters + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if niters >= N
        stop = true;
    end
    
end

end