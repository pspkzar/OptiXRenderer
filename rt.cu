#include <optix.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_math.h>
#include <optixu/optixu_vector_types.h>
#include <optixu/optixu_aabb.h>

//light properties
rtDeclareVariable(float3, lightDir, , );

//camera properties
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,         fov, , );

//ray types
rtDeclareVariable(int, Phong, ,);
rtDeclareVariable(int, Shadow, ,);

//ray payloads
struct PerRayDataRadiance{
    float4 color;
};

struct PerRayDataShadow{
    int hit;
};

rtDeclareVariable(PerRayDataRadiance, rad_res, rtPayload, );
rtDeclareVariable(PerRayDataShadow, shadow_res, rtPayload, );

//material variables
rtDeclareVariable(int, texCount, , );
rtTextureSampler<float4,2> tex0;
rtDeclareVariable(int, bumpCount, , );
rtTextureSampler<float,2> bump;
rtDeclareVariable(float4, diffuse, , );
rtDeclareVariable(float4, specular, , );
rtDeclareVariable(float, shininess, , );


//geomerty buffers
rtBuffer<float3>vertex_buffer;
rtBuffer<float3>normal_buffer;
rtBuffer<int3>index_buffer;
rtBuffer<float2>texCoord_buffer;
rtDeclareVariable(int, hasTexCoord, , );
rtBuffer<float3>tangent_buffer;
rtBuffer<float3>bitangent_buffer;

//intersection attributes
rtDeclareVariable(float2, texCoord, attribute texCoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(float3, tangent, attribute tangent, );
rtDeclareVariable(float3, bitangent, attribute bitangent, );

//ray and kernel size info
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

//output buffer
rtDeclareVariable(rtObject, top_object, , );
rtBuffer<float4,2> output0;

RT_PROGRAM void pinhole_camera(){
    float ratio=float(launch_dim.x)/float(launch_dim.y);
    float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*V*fov*ratio + d.y*U*fov + W);

	optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, Phong, 0.00000000001, RT_DEFAULT_MAX);
    PerRayDataRadiance rad_res;
    rad_res.color=make_float4(0.0f,0.0f,0.0f,0.0f);

	rtTrace(top_object, ray, rad_res);

	output0[launch_index] = rad_res.color;
	//output0[launch_index] = make_float4(1.f,0.f,0.f,0.f);
}

RT_PROGRAM void exception(){
    int code = rtGetExceptionCode();
    if(code==RT_EXCEPTION_STACK_OVERFLOW){
        output0[launch_index] = make_float4(1.f,0.f,0.f,0.f);
    }
}

RT_PROGRAM void closest_hit_radiance(){
    float4 color;

    float3 local_normal=shading_normal;
    if(bumpCount){
        float delta_x=tex2D(bump,texCoord.x+0.001,texCoord.y)-tex2D(bump,texCoord.x-0.001,texCoord.y);
        float delta_y=tex2D(bump,texCoord.x,texCoord.y+0.001)-tex2D(bump,texCoord.x,texCoord.y-0.001);
        local_normal+=7.5f*(delta_x*tangent+delta_y*bitangent);
    }

    float3 world_geo_normal=normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal=normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, local_normal));
	float3 ffnormal=faceforward(world_shade_normal, -ray.direction, world_geo_normal);

    float3 pos=ray.origin+ray.direction*t_hit;

    float intensity=fmaxf(dot(ffnormal,-lightDir),0.f);

    if(texCount>0)
    {
        color=diffuse*tex2D(tex0,texCoord.x,texCoord.y);
    }
    else
    {
        color=diffuse;
    }

    if(intensity>0){
        optix::Ray shadow_ray =optix::make_Ray(pos,-lightDir,Shadow,0.1,RT_DEFAULT_MAX);
        PerRayDataShadow prds;
        rtTrace(top_object, shadow_ray, prds);
        if(prds.hit){
            intensity*=0.3f;
        }
    }
    color*=fmaxf(intensity,0.3f);
    rad_res.color=color;
}

RT_PROGRAM void any_hit_radiance(){
    float4 color;
    if(texCount>0)
    {
        color=diffuse*tex2D(tex0,texCoord.x,texCoord.y);
    }
    else
    {
        color=diffuse;
    }
    if(color.w==0.f) rtIgnoreIntersection();
}

RT_PROGRAM void any_hit_shadow(){
    float4 color;
    if(texCount>0)
    {
        color=diffuse*tex2D(tex0,texCoord.x,texCoord.y);
    }
    else
    {
        color=diffuse;
    }
    if(color.w==0.f) rtIgnoreIntersection();
    else{
        shadow_res.hit=1;
        rtTerminateRay();
    }
}

RT_PROGRAM void miss_radiance(){
    rad_res.color=make_float4(0.f,1.f,0.f,0.f);
}

RT_PROGRAM void miss_shadow(){
    shadow_res.hit=0;
}

RT_PROGRAM void intersectMesh(int primIdx){
    //get indices
    int3 id=index_buffer[primIdx];
    //get vertices
    float3 v1=vertex_buffer[id.x];
    float3 v2=vertex_buffer[id.y];
    float3 v3=vertex_buffer[id.z];
    //intersect ray with triangle
    float3 n;
    float t, beta, gamma;
    if(intersect_triangle(ray, v1, v2, v3, n, t, beta, gamma))
    {
        if(rtPotentialIntersection(t))
        {
            //loading normals
            float3 n1=normal_buffer[id.x];
            float3 n2=normal_buffer[id.y];
            float3 n3=normal_buffer[id.z];

            float3 t1=tangent_buffer[id.x];
            float3 t2=tangent_buffer[id.y];
            float3 t3=tangent_buffer[id.z];

            float3 b1=bitangent_buffer[id.x];
            float3 b2=bitangent_buffer[id.y];
            float3 b3=bitangent_buffer[id.z];

            //loading texCoords
            if(hasTexCoord){
                float2 t1=texCoord_buffer[id.x];
                float2 t2=texCoord_buffer[id.y];
                float2 t3=texCoord_buffer[id.z];
                texCoord=(1.0f-beta-gamma)*t1 + beta*t2 +gamma*t3;
            }
            else
            {
                texCoord=make_float2(1.0f,0.0f);
            }
            //setting attributes
            shading_normal=normalize((1.0f-beta-gamma)*n1 + beta*n2 +gamma*n3);
            geometric_normal=normalize(n);
            tangent=normalize((1.0f-beta-gamma)*t1 + beta*t2 +gamma*t3);
            bitangent=normalize((1.0f-beta-gamma)*b1 + beta*b2 +gamma*b3);
            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void boundingBoxMesh(int primIdx, float result[6]){
    //get indices
    int3 id=index_buffer[primIdx];
    //load vertices
    float3 v1=vertex_buffer[id.x];
    float3 v2=vertex_buffer[id.y];
    float3 v3=vertex_buffer[id.z];
    const float area = length(cross(v2-v1,v3-v1));
    optix::Aabb* aabb = (optix::Aabb*)result;
    if(area>0.0f)
    {
        aabb->m_min=fminf(fminf(v1,v2),v3);
        aabb->m_max=fmaxf(fmaxf(v1,v2),v3);
    }
    else
    {
        aabb->invalidate();
    }
}
