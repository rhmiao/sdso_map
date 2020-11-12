root_dir='/home/mrh/Record/cgdso_test/result_origin_data/05/';
mean_time=zeros(1,2);
time_error=0;
file_dir=strcat(root_dir,'/runTimes.txt');
if exist(file_dir,'file')
    data=importdata(file_dir);
    run_time_all=data(:,1);
    run_time_all=run_time_all((run_time_all~=-1));
    mean_time(1,1)=mean(run_time_all);
end


file_dir=strcat(root_dir,'/runTimes_contrast.txt');
if exist(file_dir,'file')
    data_contra=importdata(file_dir);
    run_time_all_contra=data_contra(:,1);
    run_time_all_contra=run_time_all_contra((run_time_all_contra~=-1));
    mean_time(1,2)=mean(run_time_all_contra);
    time_error=mean_time(1,2)-mean_time(1,1);
end
figure;plot(run_time_all,'-s');
hold on;plot(run_time_all_contra,'-o');
ylabel('time(ms)');
legend('R-SDSO','SDSO');
axis([-inf inf,0,1100]);
set(gca,'FontName','Times New Roman','FontSize', 16,'FontWeight','norm');
%title('Average number of tracking points on case-05 of KITTI dataset','FontWeight','bold');
