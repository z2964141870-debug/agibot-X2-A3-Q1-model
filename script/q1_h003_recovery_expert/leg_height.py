"""Project each ankle chain onto world vertical using only IMU gravity and leg joints."""
from math import cos, sin
import xml.etree.ElementTree as ET
import numpy as np
from q1_kinematics_np import quat_wxyz_to_mat

class LegHeight:
    def __init__(self,xml):
        root=ET.parse(xml)
        self.legs=[]
        for side in ("left","right"):
            chain=[]
            for name in ("hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll"):
                body=root.find(f".//body[@name='{side}_{name}_link']")
                joint=body.find("joint")
                assert np.allclose(np.fromstring(joint.get("pos","0 0 0"),sep=" "),0.)
                axis=np.fromstring(joint.get("axis"),sep=" ")
                idx=int(abs(axis).argmax())
                assert axis[idx]==1 and np.count_nonzero(axis)==1
                pos=tuple(map(float,body.get("pos").split()))
                rotation=None
                if "quat" in body.attrib:
                    rotation=tuple(float(i) for i in quat_wxyz_to_mat(np.fromstring(body.get("quat"),sep=" ")).ravel())
                chain.append((pos,rotation,idx))
            self.legs.append(chain)

    def _project(self,up,q):
        result=[]
        for i,chain in enumerate(self.legs):
            a,b,c=map(float,up)
            z=0.
            for j,((x,y,h),rot,axis) in enumerate(chain):
                z+=a*x+b*y+c*h
                if rot is not None:
                    a,b,c=(a*rot[0]+b*rot[3]+c*rot[6],
                           a*rot[1]+b*rot[4]+c*rot[7],a*rot[2]+b*rot[5]+c*rot[8])
                cs=cos(q[i*6+j])
                sn=sin(q[i*6+j])
                if axis==0: b,c=b*cs+c*sn,c*cs-b*sn
                elif axis==1: a,c=a*cs-c*sn,c*cs+a*sn
                else: a,b=a*cs+b*sn,b*cs-a*sn
            z+=.020*a-.022*c
            result.append((z,c))
        return result

    def measure(self,up,q):
        lowest,vertical=min(self._project(up,q))
        valid=float(vertical>.65)
        return min(.65,max(0.,-lowest))*valid,valid

    def width(self,up,q):
        norm=max(1e-6,(float(up[1])**2+float(up[2])**2)**.5)
        feet=self._project((0.,up[2]/norm,-up[1]/norm),q)
        return abs(feet[0][0]-feet[1][0])

