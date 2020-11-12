%%
clear all;
close all;
format long

name='kitti_01';
dir='/home/mrh/Record/cgdso_test/panorama_result/';
A{1}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_0/dsoTracking.txt']);
B{1}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_0/gtTracking.txt']);
A{2}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_1/dsoTracking.txt']);
B{2}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_1/gtTracking.txt']);
A{3}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_2/dsoTracking.txt']);
B{3}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_2/gtTracking.txt']);
A{4}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_3/dsoTracking.txt']);
B{4}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_3/gtTracking.txt']);
A{5}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_5/dsoTracking.txt']);
B{5}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_5/gtTracking.txt']);
A{6}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_adaptive/dsoTracking.txt']);
B{6}=importdata(['/home/mrh/Record/cgdso_test/panorama_result/stereo_factor/',name,'/lambda_adaptive/gtTracking.txt']);
lambda_size=6;
posDSO=A{lambda_size}(:,2:4);
posGT=B{lambda_size}(:,2:4);
min_size=min(size(posDSO,1),size(posGT,1))-1;
figure;plot3(posGT(1:min_size,3)-posGT(1,3),posGT(1:min_size,1)-posGT(1,1),posGT(1:min_size,2)-posGT(1,2),'Linewidth',2);
for i=1:lambda_size
    posDSO=A{i}(:,2:4);
    posGT=B{i}(:,2:4);
    min_size=min(size(posDSO,1),size(posGT,1))-1;
    roataionDSO=A{i}(:,5:7);
    roatationGT=B{i}(:,5:7);
    hold on;plot3(posDSO(1:min_size,3)-posDSO(1,3),posDSO(1:min_size,1)-posDSO(1,1),posDSO(1:min_size,2)-posDSO(1,2),'Linewidth',2);
    
    if(i==lambda_size)
        axis([-inf inf,-inf,inf]);
        legend('ground truth','\lambda=0','\lambda=1','\lambda=2','\lambda=3','\lambda=5','adaptive \lambda');
        xlabel('x(m)'),ylabel('y(m)'),zlabel('z(m)','FontSize', 16);
        view(0,90);
        set(gca,'FontName','Times New Roman','FontSize', 20,'FontWeight','norm');
        saveas(gcf, [dir,name,'_pos.jpg']);
    end
 
    %figure;
    %subplot(311);plot(posDSO(1:min_size,1)-posGT(1:min_size,1),'Linewidth',3);xlabel('Time(s)');ylabel('Error(m)');
    %subplot(312);plot(posDSO(1:min_size,2)-posGT(1:min_size,2),'Linewidth',3);xlabel('Time(s)');ylabel('Error(m)');
    %subplot(313);plot(posDSO(1:min_size,3)-posGT(1:min_size,3),'Linewidth',3);xlabel('Time(s)');ylabel('Error(m)');
    %saveas(gcf, [dir,name,'_error.jpg']);
end


% figure;plot(180/pi*roataionDSO(:,1),'Linewidth',1);
% hold on;plot(180/pi*roatationGT(:,1),'Linewidth',1);legend('pos x');xlabel('time(s)');ylabel('degree(。)');
% legend('rdso','ground truth');
% figure;plot(180/pi*roataionDSO(:,2),'Linewidth',1);
% hold on;plot(180/pi*roatationGT(:,2),'Linewidth',1);legend('pos y');xlabel('time(s)');ylabel('degree(。)');
% legend('rdso','ground truth');
% figure;plot(180/pi*roataionDSO(:,3),'Linewidth',1);
% hold on;plot(180/pi*roatationGT(:,3),'Linewidth',1);legend('pos z');xlabel('time(s)');ylabel('degree(。)');
% legend('rdso','ground truth');
% 
% figure;
% subplot(311);plot(180/pi*(roataionDSO(1:min_size,1)-roatationGT(1:min_size,1)),'Linewidth',1);xlabel('Time(s)');ylabel('Error(degree)');
% subplot(312);plot(180/pi*(roataionDSO(1:min_size,2)-roatationGT(1:min_size,2)),'Linewidth',1);xlabel('Time(s)');ylabel('Error(degree)');
% subplot(313);plot(180/pi*(roataionDSO(1:min_size,3)-roatationGT(1:min_size,3)),'Linewidth',1);xlabel('Time(s)');ylabel('Error(degree)');


%%
% bar(l,'DisplayName','l')
% legend('okvis','rovio','vins mono','s-msckf','our method');title('RMSE on EuRoC dataset','FontWeight','bold');
% ylabel('RMSE[m]');
% set(gca,'xticklabel',{'V1-01','V1-02','V1-03','V2-01','V2-02','V2-03','MH-03','MH-04','MH-05'});
% set(gca,'FontName','Times New Roman','FontSize', 14,'FontWeight','norm');
% grid on;
