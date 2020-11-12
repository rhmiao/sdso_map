close all;
root_dir='/home/mrh/Record/cgdso_test/wideFoV_tracking/assist_feature/kitti_';
mean_time=zeros(10,2);
mean_pos=zeros(10,2);
mean_rot=zeros(10,2);
time_error=zeros(10,1);
points_error=zeros(10,1);
for i=10:10
    data_dir=strcat(root_dir,sprintf('%02d',i));
    if exist(data_dir,'dir')
        file_dir=strcat(data_dir,'/trackTimeAndPtsNum.txt');
        if exist(file_dir,'file')
            data=importdata(file_dir);
            track_time_all=data(:,1);
            track_pos_all=sum(abs(data(:,3:5)).^2,2).^(1/2);
            track_rot_all=sum(abs(data(:,6:8)).^2,2).^(1/2);
            track_time=track_time_all((track_time_all~=-1));
            track_pos=track_pos_all((track_pos_all~=0));
            track_rot=track_rot_all((track_rot_all~=0));
            mean_time(i+1,1)=mean(track_time);
            mean_pos(i+1,1)=mean(track_pos);
            mean_rot(i+1,1)=mean(track_rot);
        end
        
        
        file_dir=strcat(data_dir,'/contrast/trackTimeAndPtsNum.txt');
        if exist(file_dir,'file')
            data_contra=importdata(file_dir);
            track_time_all_contra=data_contra(:,1);
            track_pos_all_contra=sum(abs(data_contra(:,3:5)).^2,2).^(1/2);
            track_rot_all_contra=sum(abs(data_contra(:,6:8)).^2,2).^(1/2);
            track_time_contra=track_time_all_contra((track_time_all_contra~=-1));
            track_pos_contra=track_pos_all_contra((track_pos_all_contra~=0));
            track_rot_contra=track_rot_all_contra((track_rot_all_contra~=0));
            mean_time(i+1,2)=mean(track_time_contra);
            mean_pos(i+1,2)=mean(track_pos_contra);
            mean_rot(i+1,2)=mean(track_rot_contra);
        end
    end
end
figure;plot(0:9,mean_time(:,1),'-s');
hold on;plot(0:9,mean_time(:,2),'-o');
ylabel('Average tracking time (ms)');
legend('with virtual wide FoV','without virtual wide FoV');
%axis([-inf inf,0,1100]);
set(gca,'xticklabel',{'V1\_01','V1\_02','V1\_03','V2\_01','V2\_02','MH\_01','MH\_02','MH\_03','MH\_04','MH\_05'});
set(gca,'FontName','Times New Roman','FontSize', 16,'FontWeight','norm');
title('Average tracking time on EuRoC dataset','FontWeight','bold');

figure;plot(0:9,mean_pos(:,1),'-s');
hold on;plot(0:9,mean_pos(:,2),'-o');
ylabel('Average tracking position error (m)');
legend('with virtual wide FoV','without virtual wide FoV');
set(gca,'xticklabel',{'V1\_01','V1\_02','V1\_03','V2\_01','V2\_02','MH\_01','MH\_02','MH\_03','MH\_04','MH\_05'});
set(gca,'FontName','Times New Roman','FontSize', 16,'FontWeight','norm');
% axis([-inf inf,0,1100]);
title('Average tracking position error of each frame on EuRoC dataset','FontWeight','bold');

figure;plot(0:9,mean_rot(:,1),'-s');
hold on;plot(0:9,mean_rot(:,2),'-o');
ylabel('Average tracking rotation error (rad)');
legend('with virtual wide FoV','without virtual wide FoV');
set(gca,'xticklabel',{'V1\_01','V1\_02','V1\_03','V2\_01','V2\_02','MH\_01','MH\_02','MH\_03','MH\_04','MH\_05'});
set(gca,'FontName','Times New Roman','FontSize', 16,'FontWeight','norm');
% axis([-inf inf,0,1100]);
title('Average tracking rotation error of each frame on EuRoC dataset','FontWeight','bold');
