#include "context.h"


rtDeclareVariable(float4, Kd, , ); //diffuse coefficient
rtDeclareVariable(float4, Ks, , ); //specular coefficient
rtDeclareVariable(float, Ns, , ); // shininess



rtTextureSampler<float4, 2> map_Kd; //diffuse texture
