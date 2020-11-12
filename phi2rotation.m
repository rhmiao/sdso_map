function [R]=phi2rotation(phi)
    new_phi=[];
    if (size(phi,1)<3)
        new_phi=phi;
    else
        new_phi=phi';
    end
    theta=norm(new_phi);
    axang = [new_phi theta];
    R = axang2rotm(axang);
end