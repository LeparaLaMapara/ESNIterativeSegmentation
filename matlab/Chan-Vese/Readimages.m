function Readimages(action, method)



% if action =='train' 
%     path_destination = 'F:\MSC\Data\FINAL_DATA\HETEROGENEOUS\WEIZMANN\SEGMENTATION_DATA\BINARY_SEGMENTATION\TRAIN\';
%     path_input       = 'F:\MSC\Data\FINAL_DATA\HETEROGENEOUS\WEIZMANN\TRAIN\';    
%    path_destination = 'F:\MSC\Data\BERKELEY_DATA\TIME_SERIES_BERKELEY\SEGMENTATION_DATA\BINARY_SEGMENTATION\TRAIN\';
%    path_input       = 'F:\MSC\Data\BERKELEY_DATA\TIME_SERIES_BERKELEY\TRAIN\';
    
% if action=='test'
%     path_destination = 'F:\MSC\Data\FINAL_DATA\HETEROGENEOUS\WEIZMANN\SEGMENTATION_DATA\BINARY_SEGMENTATION\val\';
%     path_input       = 'F:\MSC\Data\FINAL_DATA\HETEROGENEOUS\WEIZMANN\val\';   
%     path_destination = 'F:\MSC\Data\FINAL_DATA\HOMOGENEOUS\SATELLITE_DATA\SEGMENTATION\BINARY_SEGMENTATION\test\';
%     path_input       = 'F:\MSC\Data\FINAL_DATA\HOMOGENEOUS\SATELLITE_DATA\test\';   
%     path_destination = 'F:\MSC\Data\BERKELEY_DATA\TIME_SERIES_BERKELEY\SEGMENTATION_DATA\BINARY_SEGMENTATION\TEST\';
%     path_input       = 'F:\MSC\Data\BERKELEY_DATA\TIME_SERIES_BERKELEY\TEST\';
% else 
% 
% %     path_destination = 'F:\MSC\Data\SATELLITE_DATA\TIME_SERIES_DATA\SEGMENTATION\BINARY_SEGMENTATION\VAL\';
% %     path_input       = 'F:\MSC\Data\SATELLITE_DATA\TIME_SERIES_DATA\VAL\';   
%     path_destination = 'F:\MSC\Data\BERKELEY_DATA\TIME_SERIES_BERKELEY\SEGMENTATION_DATA\BINARY_SEGMENTATION\VAL\';
%     path_input       = 'F:\MSC\Data\BERKELEY_DATA\TIME_SERIES_BERKELEY\VAL\';
    
    
% end



datasets =  {'CIFAR_100', 'CIFAR_10' }%{'BSR','WEIZMANN', 'COCCO'} ;
for data = 1:length(datasets) 
    ds = datasets{data};
    path_destination = strcat('F:\MSC\Data\processed_data\',ds,'\labels\');
    path_input       = strcat('F:\MSC\Data\processed_data\', ds,'\images\');   

%     display(ds)
    if strcmpi('WEIZMANN',ds)
        file_extention='.png';
    else
        file_extention='.jpg';

        
    display(path_destination)
    files = dir (strcat(path_input,'*',file_extention));
    L = length (files); % number of original images
    display(L)
    for i=1:L
        image_name = strcat(path_input,string(i),file_extention);
        display(image_name);
        seg = chenvese(image_name, path_destination,'whole', 100, 0.1, method);
    end
    end
end
end



