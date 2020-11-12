%%
clear all;
close all;
format long

name='lambda_adaptive';
kitti_name='kitti_';
root_dir='/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/kitti_';
gt_path='/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/kitti_kit/data_odometry_poses_rdso/';
dso_path='/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/kitti_kit/results_rdso/data/';
for i=1:11
    gt_file=[root_dir,num2str(i-1,'%02d'),'/',name,'/dsoTracking.txt'];
    dso_file=[root_dir,num2str(i-1,'%02d'),'/',name,'/gtTracking.txt'];
    if(exist(gt_file))
        copyfile(gt_file,[gt_path,num2str(i-1,'%02d'),'.txt']);
        copyfile(dso_file,[dso_path,num2str(i-1,'%02d'),'.txt']);
    end
end
%%
% bar(l,'DisplayName','l')
% legend('okvis','rovio','vins mono','s-msckf','our method');title('RMSE on EuRoC dataset','FontWeight','bold');
% ylabel('RMSE[m]');
% set(gca,'xticklabel',{'V1-01','V1-02','V1-03','V2-01','V2-02','V2-03','MH-03','MH-04','MH-05'});
% set(gca,'FontName','Times New Roman','FontSize', 14,'FontWeight','norm');
% grid on;
