#ifndef _GEOMETRY_H
#define _HEOMETRY_H


#include "context.h"

rtBuffer<int3> index_buffer;
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;

rtBuffer<float3> tangent_buufer;
rtBuffer<float3> bitangent_buffer;

rtBuffer<float2> texCoord_buffer;
rtDeclareVariable(int, hasTexCoord, , );

#endif // _GEOMETRY_H
