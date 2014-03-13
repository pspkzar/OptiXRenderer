#ifndef _MATERIAL_H
#define _MATERIAL_H


#include "context.h"


rtDeclareVariable(float4, Kd, , ); //diffuse coefficient
rtDeclareVariable(float4, Ks, , ); //specular coefficient
rtDeclareVariable(float, Ns, , ); // shininess

rtTextureSampler<float4, 2> map_Kd; //diffuse texture
rtTextureSampler<float4, 2> map_Ks; //specular texture
rtTextureSampler<float, 2> map_bump; //bump map

#endif //_MATERIAL_H
