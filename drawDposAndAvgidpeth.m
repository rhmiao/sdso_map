%%
clear all;
close all;
file_dir='/home/mrh/catkin_ws/src/cgdso/logs/trackDeltaPosAndAvgidepth.txt';
data=importdata(file_dir);
sizen=size(data,1);

figure;hold on;
[AX,H1,H2]=plotyy(1:sizen,data(:,5),1:sizen,2*data(:,3));
plot(1:sizen,4*data(:,4),'g','LineWidth',2);
plot(1:sizen,ones(sizen,1)*2,'b--','LineWidth',2);
%[AX,H1,H2]=plotyy(1:sizen,0.5+3*data(:,3),[[1:sizen]',[1:sizen]'],[data(:,4)',ones(sizen)'*2]);
pos1=get(AX(1),'position');
xlim1 = get(AX(1),'xlim');
set(get(AX(1),'Ylabel'),'String','Inverse depth (m^{-1})')
set(get(AX(2),'Ylabel'),'String','Delta position z (m)')
set(H1,'Linewidth',2)
set(H2,'Linewidth',2)
set(AX(1),'Ylim',[0,3])
set(AX(1),'yTick',[0:1:3])
set(AX(2),'Ylim',[0,0.6])
set(AX(2),'yTick',[0:0.2:0.6])
xlabel('Frames','FontSize', 16)
set(gca,'FontName','Times New Roman','FontSize', 16,'FontWeight','norm');
set(AX(2),'FontName','Times New Roman','FontSize', 16,'FontWeight','norm');
title('The relationship between camera motion and inverse depth','FontWeight','bold');
legend('average inverse depth','delta roatation','threshold','delta z');
pos1(1)=pos1(1);
pos1(3) = pos1(3)*0.9;
set([AX(1);AX(2)],'position',pos1);
pos3 = pos1;
pos3(3) = pos3(3)+0.1;
xlim3 = xlim1;
xlim3(2) = xlim3(1)+(xlim1(2)-xlim1(1))/pos1(3)*pos3(3);
AX(3) = axes('position',pos3, 'color','none','ycolor',[0,0.5,0],'xlim',xlim3, ...
    'xtick',[],'yTick',[0:0.1:0.3],'Ylim',[0,0.3],'yaxislocation','right','yminortick','off');
ylim3 = get(AX(3), 'ylim');
line([xlim1(2),xlim3(2)],[ylim3(1),ylim3(1)],'parent',AX(3),'color',[1,1,1],'LineWidth',1.5);
set(get(AX(3),'Ylabel'),'String','Delta rotation (rad)')
set(AX(3),'FontName','Times New Roman','FontSize', 16,'FontWeight','norm');



