function getBSRdata(path)
% getting the segmetation images from the Bersklry dataset.

myDir  = strcat('F:\MSC\Data\BSR\BSDS500\data\groundTruth\',path);
myFiles = dir(fullfile(myDir,'*.mat'));
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  file  = open(fullfile(myDir, baseFileName));
 
  image = file.groundTruth{2}.Boundaries;
  newname= split(baseFileName,".");
  g= cell2mat (newname(1))
  
  imwrite(image,strcat('F:\MSC\Data\BSR\BSDS500\data\groundtruthimages\', path,'\',g, '.png'))
  %imshow(image)
end
end