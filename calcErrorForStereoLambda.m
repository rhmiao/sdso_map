%%
clear all;
close all;
format long

error_path='/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/kitti_kit/results_rdso/errors/';
for i=1:11
    error_file=[error_path,num2str(i-1,'%02d'),'.txt'];
    if(exist(error_file))
        A=importdata(error_file);
        str=sprintf('%s:%f',num2str(i-1,'%02d'),mean(A(:,3)));
        disp(str)
    end
end
