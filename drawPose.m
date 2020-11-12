%%
clear all;
close all;
format long

% theta_x=0.3;
% theta_y=0.7;
% theta_z=-4.8;
theta_x=-46;
theta_y=-10;
theta_z=35;
Rx=[    1   0    0;
    0    cos(theta_x*pi/180)   sin(theta_x*pi/180);
    0    -sin(theta_x*pi/180)    cos(theta_x*pi/180)];
Ry=[    cos(theta_y*pi/180)   0    sin(theta_y*pi/180);
    0    1   0;
    -sin(theta_y*pi/180)    0    cos(theta_y*pi/180)];
Rz=[    cos(theta_z*pi/180)   sin(theta_z*pi/180)    0;
    -sin(theta_z*pi/180)    cos(theta_z*pi/180)   0;
    0    0    1];
Rbase=[0.9992   -0.0264    0.0284;
    0.0144    0.9325    -0.3609;
   -0.0360   0.3602    0.9322];
Rleica=[0  0    -1;
    0    -1    0;
   1   0    0];%01-02
%R=Rz*Ry*Rx*Rbase;
%R=Rz*Ry*Rx*Rleica;
R=eye(3);
% Rtra=[ 0.3597684732  0.5816363768 -0.7295654671;
% -0.0261950885  0.7879098702  0.6152331702;
%  0.9326738246 -0.2022304663  0.2987011474];
name='vh02_01';
%dir='/home/mrh/Record/cgdso_test/kitti_result/';
%A=importdata(['/home/mrh/Record/cgdso_test/result_origin_data/dsoTracking_',name,'.txt']);
%B=importdata(strcat('/home/mrh/Record/cgdso_test/result_origin_data/gtTracking_',name,'.txt'));
dir='/home/mrh/catkin_ws/src/cgdso/logs/';
A=importdata(['/home/mrh/Record/cgdso_test/euroc_result/SDSO/',name,'/dsoTracking.txt']);
B=importdata(['/home/mrh/Record/cgdso_test/euroc_result/SDSO/',name,'/gtTracking.txt']);
% posDSO=(1/1)*(Rtra'*A(:,2:4)')';
posDSO=A(:,2:4);
posGT=(R*B(:,2:4)')';
trans=[B(:,1) posGT B(:,5:7)];
%A_cont=importdata(['/home/mrh/Record/cgdso_test/wideFoV_tracking/assist_feature/',name,'/contrast/dsoTracking.txt']);
% posDSO_cont=A_cont(:,2:4);
%dlmwrite(strcat(dir,'gtTracking_new.txt')',trans,'delimiter','\t','precision','%.6f') 
min_size=min(size(posDSO,1),size(posGT,1))-100;
seg_length=10;
seg_start=1:seg_length:min_size;
seg_start_size=size(seg_start,2);
roataionDSO=A(:,5:7);
roatationGT=(R*B(:,5:7)')';

disGT=[0];
disTmp=0;
single_length=100;
for i=2:min_size
    disTmp=disTmp+norm(posGT(i,:)-posGT(i-1,:));
    disGT=[disGT disTmp];
end
segNum=min(floor(disTmp/single_length),8);

figure;plot3(posDSO(1:min_size,3),posDSO(1:min_size,1),posDSO(1:min_size,2),'Linewidth',3);
%figure;plot3(-posDSO(:,1),posDSO(:,3),-posDSO(:,2),'Linewidth',1);
%hold on;plot3(posDSO_cont(:,3),posDSO_cont(:,1),posDSO_cont(:,2),'Linewidth',1);
hold on;plot3(posGT(1:min_size,3),posGT(1:min_size,1),posGT(1:min_size,2),'Linewidth',3);
legend('R-SDSO','ground truth');
xlabel('x(m)'),ylabel('y(m)'),zlabel('z(m)','FontSize', 20);
set(gca,'FontName','Times New Roman','FontSize', 20,'FontWeight','norm');
view(0,90)
%saveas(gcf, [dir,name,'_pos.jpg']);

% figure;plot(posDSO(:,1),'Linewidth',1);
% hold on;plot(posGT(:,1),'Linewidth',1);legend('pos x');xlabel('time(s)');ylabel('pos(m)');
% legend('rdso','ground truth');
% figure;plot(posDSO(:,2),'Linewidth',1);
% hold on;plot(posGT(:,2),'Linewidth',1);legend('pos y');xlabel('time(s)');ylabel('pos(m)');
% legend('rdso','ground truth');
% figure;plot(posDSO(:,3),'Linewidth',1);
% hold on;plot(posGT(:,3),'Linewidth',1);legend('pos z');xlabel('time(s)');ylabel('pos(m)');
% legend('rdso','ground truth');
 
figure;
subplot(311);plot(posDSO(1:min_size,1)-posGT(1:min_size,1),'Linewidth',3);xlabel('Time(s)');ylabel('Error(m)');
subplot(312);plot(posDSO(1:min_size,2)-posGT(1:min_size,2),'Linewidth',3);xlabel('Time(s)');ylabel('Error(m)');
subplot(313);plot(posDSO(1:min_size,3)-posGT(1:min_size,3),'Linewidth',3);xlabel('Time(s)');ylabel('Error(m)');
%saveas(gcf, [dir,name,'_error.jpg']);

errorRV=zeros(1,segNum);
errorTV=zeros(1,segNum);
errorRmat=[];
 errorTmat=[];
 for start_size=1:seg_start_size
     startframe=seg_start(start_size);
     startRGT=phi2rotation(roatationGT(startframe,:));
     startRDSO=phi2rotation(roataionDSO(startframe,:));
     startTGT=posGT(startframe,:);
     startTDSO=posDSO(startframe,:);
     dis=0;
     lastframe=startframe+1;
     if(disGT(min_size)-disGT(startframe)<single_length*segNum)
         continue;
     end
     for num=1:segNum
         while(lastframe<min_size && disGT(lastframe)-disGT(startframe)<num*single_length)
             lastframe=lastframe+1;
         end
         deltaD=disGT(lastframe)-disGT(startframe);
 
         lastRGT=phi2rotation(roatationGT(lastframe,:));
         lastRDSO=phi2rotation(roataionDSO(lastframe,:));
         lastTGT=posGT(lastframe,:);
         lastTDSO=posDSO(lastframe,:);
         
         delta_RGT=lastRGT/startRGT;
         delta_TGT=(lastTGT-startTGT)/startRGT;
         
         delta_RDSO=lastRDSO/startRDSO;
         delta_TDSO=(lastTDSO-startTDSO)/startRDSO;
         error_R=180/pi*real(acos((trace(inv(delta_RGT)*delta_RDSO)-1)/2))/deltaD;
         error_T=100*norm(delta_TDSO-delta_TGT)/deltaD;
         errorRV(num)=error_R;
         errorTV(num)=error_T;
     end
     errorRmat=[errorRmat;errorRV];
     errorTmat=[errorTmat;errorTV];
 end
 mean_errorR=mean(errorRmat);
 mean_errorT=mean(errorTmat);
 figure;
 subplot(211);plot(single_length:single_length:single_length*size(mean_errorT,2),mean_errorT,'s-','Linewidth',2);xlabel('Path Length[m]');ylabel('Translation Error[%]');
 legend('Translation Error');
 subplot(212);plot(single_length:single_length:single_length*size(mean_errorT,2),mean_errorR,'s-','Linewidth',2);xlabel('Path Length[m]');ylabel('Rotation Error[deg/m]');
 legend('Rotation Error');
set(gca,'FontName','Times New Roman','FontSize', 20,'FontWeight','norm');
% saveas(gcf, [dir,name,'_stand_error.jpg']);
% 
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

t_e1=sqrt(mean((posDSO(1:20:min_size,1)-posGT(1:20:min_size,1)).^2))
t_e2=sqrt(mean((posDSO(1:20:min_size,2)-posGT(1:20:min_size,2)).^2))
t_e3=sqrt(mean((posDSO(1:20:min_size,3)-posGT(1:20:min_size,3)).^2))
t_e4=sqrt(t_e1^2+t_e2^2+t_e3^2)

r_e1=sqrt(mean(((roataionDSO(1:min_size-1,1)-roatationGT(1:min_size-1,1))).^2))
r_e2=sqrt(mean(((roataionDSO(1:min_size-1,2)-roatationGT(1:min_size-1,2))).^2))
r_e3=sqrt(mean(((roataionDSO(1:min_size-1,3)-roatationGT(1:min_size-1,3))).^2))
r_e4=sqrt(r_e1^2+r_e2^2+r_e3^2)
% s_error_min=1e10;
% theta_x_min=0;
% theta_y_min=0;
% theta_z_min=0;
% for theta_x=-60:0.2:-40
%     for theta_y=-20:0.2:0
%         for theta_z=20:0.2:40
%             Rx=[    1   0    0;
%                 0    cos(theta_x*pi/180)   sin(theta_x*pi/180);
%                 0    -sin(theta_x*pi/180)    cos(theta_x*pi/180)];
%             Ry=[    cos(theta_y*pi/180)   0    sin(theta_y*pi/180);
%                 0    1   0;
%                 -sin(theta_y*pi/180)    0    cos(theta_y*pi/180)];
%             Rz=[    cos(theta_z*pi/180)   sin(theta_z*pi/180)    0;
%                 -sin(theta_z*pi/180)    cos(theta_z*pi/180)   0;
%                 0    0    1];
%             R=Rz*Ry*Rx*Rleica;
%             roatationGT=(R*B(:,5:7)')';
%             posGT=(R*B(:,2:4)')';
%             s_error=0;
%             for i=1:3
%                 s_error=s_error+sum(posDSO(1:min_size,i)-posGT(1:min_size,i)).^2;
%             end
%             if(s_error<s_error_min)
%                 s_error_min=s_error;
%                 theta_x_min=theta_x;
%                 theta_y_min=theta_y;
%                 theta_z_min=theta_z;
%             end
%         end
%     end
%     theta_x
% end
% theta_x_min
% theta_y_min
% theta_z_min

%%
% bar(l,'DisplayName','l')
% legend('okvis','rovio','vins mono','s-msckf','our method');title('RMSE on EuRoC dataset','FontWeight','bold');
% ylabel('RMSE[m]');
% set(gca,'xticklabel',{'V1-01','V1-02','V1-03','V2-01','V2-02','V2-03','MH-03','MH-04','MH-05'});
% set(gca,'FontName','Times New Roman','FontSize', 14,'FontWeight','norm');
% grid on;
