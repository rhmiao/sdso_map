root_dir='/home/mrh/Record/cgdso_test/result_origin_data/';
mean_time=zeros(11,2);
mean_points=zeros(11,2);
time_error=zeros(11,1);
points_error=zeros(11,1);
for i=0:10
    data_dir=strcat(root_dir,sprintf('%02d',i));
    if exist(data_dir,'dir')
        file_dir=strcat(data_dir,'/trackTimeAndPtsNum.txt');
        if exist(file_dir,'file')
            data=importdata(file_dir);
            track_time_all=data(:,1);
            points_num_all=data(:,2);
            track_time=track_time_all((track_time_all~=-1));
            points_num=points_num_all((points_num_all~=-1));
            mean_time(i+1,1)=mean(track_time);
            mean_points(i+1,1)=mean(points_num);
        end
        
        
        file_dir=strcat(data_dir,'/trackTimeAndPtsNum_contrast.txt');
        if exist(file_dir,'file')
            data_contra=importdata(file_dir);
            track_time_all_contra=data_contra(:,1);
            points_num_all_contra=data_contra(:,2);
            track_time_contra=track_time_all_contra((track_time_all_contra~=-1));
            points_num_contra=points_num_all_contra((points_num_all_contra~=-1));
            mean_time(i+1,2)=mean(track_time_contra);
            time_error(i+1,1)=mean_time(i+1,2)-mean_time(i+1,1);
            mean_points(i+1,2)=mean(points_num_contra);
            points_error(i+1,1)=mean_points(i+1,2)-mean_points(i+1,1);
        end
    end
end
figure;plot(0:10,mean_points(:,1),'-s');
hold on;plot(0:10,mean_points(:,2),'-o');
ylabel('Average number of points');
legend('with depth initialization','without depth initialization');
axis([-inf inf,0,1100]);
set(gca,'xticklabel',{'case-00','case-01','case-02','case-03','case-04','case-05','case-06','case-07','case-08','case-09','case-10'});
set(gca,'FontName','Times New Roman','FontSize', 16,'FontWeight','norm');
title('Average number of tracking points on KITTI dataset','FontWeight','bold');

figure;plot(0:10,mean_time(:,1),'-s');
hold on;plot(0:10,mean_time(:,2),'-o');
ylabel('Average tracking time');
legend('with depth initialization','without depth initialization');
set(gca,'xticklabel',{'case-00','case-01','case-02','case-03','case-04','case-05','case-06','case-07','case-08','case-09','case-10'});
set(gca,'FontName','Times New Roman','FontSize', 16,'FontWeight','norm');
% axis([-inf inf,0,1100]);
title('Average tracking time of each frame on KITTI dataset','FontWeight','bold');
