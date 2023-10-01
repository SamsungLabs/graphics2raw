% Copyright (c) 2023 Samsung Electronics Co., Ltd.

% Author(s):
% Abhijith Punnappurath (abhijith.p@samsung.com)

% Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
% Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
% "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and limitations under the License.
% For conditions of distribution and use, see the accompanying LICENSE.md file.

% Code to mask out the color chart, resize and save as PNG files 

% CHANGE mainpath AND gtmatpath TO RUN THIS CODE

clc; 
clear; 
close all;

% Below are the nine cameras in the NUS dataset
% Code assumes that 
% (1) the demosaiced linear-RGB .tif files are in
% mainpath/<camera_name>/RAW/stage4 folders
% (2) the rendered sRGB images are in
% mainpath/<camera_name>/RAW/stage11 folders
% (3) ground truth mat files for all cameras are in gtmatpath
% Note: Renamed NikonD40_RAW to NikonD40_RAW.zip

mainpath='.\';
gtmatpath='.\all_gt_illum_mat\'; 

namecell{1,1}='Canon1DsMkIII_RAW.zip';
namecell{1,2}='Canon600D_RAW.zip';
namecell{1,3}='FujifilmXM1_RAW.zip';
namecell{1,4}='NikonD40_RAW.zip';
namecell{1,5}='NikonD5200_RAW.zip';
namecell{1,6}='OlympusEPL6_RAW.zip';
namecell{1,7}='PanasonicGX1_RAW.zip';
namecell{1,8}='SamsungNX2000_RAW.zip';
namecell{1,9}='SonyA57_RAW.zip';

% There is a problem with the DNG orientation tag for the Canon1DsMkIII_RAW
% The following images need to be manually rotated so that the mask can be
% correctly applied
rotated_imgs=[5,12,48,49,52,53,54,58,60,61,62,67,75,84,87, ...
105,112,115,116,119,122, ...
124,126, ... 
131,133,137,138,144,148,149, ...
151,152,153,155,160,161,163,168,169, ...
172,181,187,189,190,191,196,200, ...
204,207,208,215,216,217,218,220,238, ...
242,245,256];

for jj=1:length(namecell)

campath=[namecell{1,jj} '\RAW\stage4'];
matpath=[gtmatpath namecell{1,jj}(1:end-8) '_gt.mat'];

savepath = fullfile(mainpath,campath,'downsampled');
mkdir(savepath)
load(matpath)
b=100; % extra border around the masks

allraw=dir(fullfile(mainpath,campath,'*.tif'));

for i=1:length(allraw)
    if(strcmp(allraw(i).name(1:end-8),all_image_names{i})~=1)
       fprintf('error \n')
       break;
    end
       
    fprintf('%s \n',allraw(i).name)
    img = imread(fullfile(mainpath,campath,allraw(i).name));
    if jj>1 %        for all cameras other than Canon1DsMkIII_RAW
        imgsrgb=imread(fullfile(mainpath,[campath(1:end-1) '11'],[allraw(i).name(1:end-8) '_st11.tif']));
        if(size(imgsrgb,1)>size(imgsrgb,2))
           img=imrotate(img,-90); 
        end
    else
       if(ismember(i,rotated_imgs))
          img=imrotate(img,-90); 
       end
    end

       
    if jj == 3 %  for fuji, mult by 3    
        img(max(CC_coords(i,1)*3-b,1):min(CC_coords(i,2)*3+b,size(img,1)),max(CC_coords(i,3)*3-b,1):min(CC_coords(i,4)*3+b,size(img,2)),:)=0;
    else % mult by 2
        img(max(CC_coords(i,1)*2-b,1):min(CC_coords(i,2)*2+b,size(img,1)),max(CC_coords(i,3)*2-b,1):min(CC_coords(i,4)*2+b,size(img,2)),:)=0;
    end
    img = imresize(img,0.2,'bicubic');
    imwrite(img,fullfile(savepath,[allraw(i).name(1:end-4) '.png']))
end    

end