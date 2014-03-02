#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <IL/il.h>
#include <IL/ilu.h>

#include <assimp/Importer.hpp>
#include <assimp/cimport.h>
#include <assimp/material.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

#define ANISOTROPY 16.0f
#define MIPMAPS 1

#define STEP 2
#define ANG_STEP 0.1

#define USE_MS 1

unsigned int LoadFlags = aiProcessPreset_TargetRealtime_MaxQuality|aiProcess_RemoveRedundantMaterials|aiProcess_PreTransformVertices;

enum EntryPoints {
    ENTRY_PINHOLE,
    ENTRY_PINHOLE_MS,
    ENTRY_COUNT
};

using namespace optix;

int width=720;
int height=720;

Context renderer;
Buffer out;

float3 eye=make_float3(0.f, 0.f, 0.f);
float3 up=make_float3(0.f,1.f,0.f);
float3 lookDir=normalize(make_float3(0.f, 0.f, -1.f));

Assimp::Importer importer;
std::string scene_p="crytek-sponza/";
std::string scene_name="sponza.obj";
std::string ptx_p="rt.ptx";

enum ray_types
{
    Shadow,
    Phong,
    RAY_TYPE_COUNT,
};

Buffer genOutputBuffer()
{
    RTformat format = RT_FORMAT_FLOAT4;
    Buffer outBuffer=renderer->createBuffer(RT_BUFFER_OUTPUT,format,width,height);
    outBuffer->validate();
    return outBuffer;
}

void reshape(int w, int h)
{
    width=w;
    height=h;
    glViewport(0,0,w,h);
    //Pass Arguments to Optix
    out->setSize(w,h);

}

inline void optix_draw()
{
    renderer->launch(USE_MS,width,height);
    void *pixels=out->map();
    glDrawPixels(width,height,GL_RGBA,GL_FLOAT,pixels);
    out->unmap();
}

void renderScene()
{
    //Render with Optix
    //clear buffer
    glClear(GL_COLOR_BUFFER_BIT);
    //optix
    optix_draw();
    //fps count
    static int frame=0, time,timebase=0;

    frame++;
    time=glutGet(GLUT_ELAPSED_TIME);
    if(time-timebase){
        float fps=frame*1000.0/(time-timebase);
        timebase=time;
        frame=0;

        std::cout<<fps<<" FPS"<<std::endl;
    }
    //swap buffers
    glutSwapBuffers();
}

inline const aiScene* loadScene(std::string scene_path)
{
    std::ifstream scene_file(scene_path.c_str());
    if(!scene_file.fail())
    {
        scene_file.close();
    }
    else
    {
        std::cout<<"Error reading scene file."<<std::endl;
        return NULL;
    }

    const aiScene *s=aiImportFile(scene_path.c_str(), LoadFlags);
    aiApplyPostProcessing(s, aiProcess_CalcTangentSpace);
    aiApplyPostProcessing(s, aiProcess_OptimizeGraph);

    if(!s)
    {
        std::cout<<"Failed to load scene: "<<importer.GetErrorString()<<std::endl;
        return NULL;
    }
    std::cout<<"Scene loaded sccessfully."<<std::endl;

    return s;
}

TextureSampler newTexture(std::string name)
{
    ILuint image=iluGenImage();
    ilBindImage(image);
    ilEnable(IL_ORIGIN_SET);
    ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
    iluBuildMipmaps();

    TextureSampler res=renderer->createTextureSampler();;
    res->setArraySize(1);

    res->setWrapMode(0,RT_WRAP_REPEAT);
    res->setWrapMode(1,RT_WRAP_REPEAT);
    res->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    res->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    res->setMaxAnisotropy(ANISOTROPY);
    res->setFilteringModes(RT_FILTER_LINEAR,RT_FILTER_LINEAR,RT_FILTER_NONE);

    ILboolean success=ilLoadImage((ILstring)(scene_p+name).c_str());
    if(success){
        ilConvertImage(IL_RGBA,IL_UNSIGNED_BYTE);
        std::vector<Buffer> mipmaps;
        int nmipmap=0;
        while(ilActiveMipmap(nmipmap)&&nmipmap<MIPMAPS){
            int w=ilGetInteger(IL_IMAGE_WIDTH);
            int h=ilGetInteger(IL_IMAGE_HEIGHT);
            //std::cout<<w<<'x'<<h<<std::endl;
            void * data= (void*)ilGetData();
            mipmaps.push_back(renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_UNSIGNED_BYTE4,w,h));
            void * dataMap = mipmaps[nmipmap]->map();
            ILint size = ilGetInteger(IL_IMAGE_SIZE_OF_DATA);
            //std::cout<<size<<std::endl;
            memcpy(dataMap,data,size);
            mipmaps[nmipmap]->unmap();
            mipmaps[nmipmap]->validate();
            nmipmap++;
            ilBindImage(image);
            iluBuildMipmaps();
        }

        res->setMipLevelCount(nmipmap);
        for(int i=0; i<nmipmap; i++){
            res->setBuffer(0,i,mipmaps[i]);
        }

        res->validate();
    }
    else{
        res->destroy();
        std::cout<<"Error reading texture: "<<name<<std::endl;
        return NULL;
    }
    ilBindImage(0);
    ilDeleteImage(image);
    return res;
}

TextureSampler newTextureBump(std::string name)
{
    ILuint image=iluGenImage();
    ilBindImage(image);
    ilEnable(IL_ORIGIN_SET);
    ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
    iluBuildMipmaps();

    TextureSampler res=renderer->createTextureSampler();;
    res->setArraySize(1);

    res->setWrapMode(0,RT_WRAP_REPEAT);
    res->setWrapMode(1,RT_WRAP_REPEAT);
    res->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    res->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    res->setMaxAnisotropy(ANISOTROPY);
    res->setFilteringModes(RT_FILTER_LINEAR,RT_FILTER_LINEAR,RT_FILTER_NONE);

    ILboolean success=ilLoadImage((ILstring)(scene_p+name).c_str());
    if(success){
        ilConvertImage(IL_LUMINANCE,IL_UNSIGNED_BYTE);
        std::vector<Buffer> mipmaps;
        int nmipmap=0;
        while(ilActiveMipmap(nmipmap)&&nmipmap<MIPMAPS){
            int w=ilGetInteger(IL_IMAGE_WIDTH);
            int h=ilGetInteger(IL_IMAGE_HEIGHT);
            //std::cout<<w<<'x'<<h<<std::endl;
            void * data= (void*)ilGetData();
            mipmaps.push_back(renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_UNSIGNED_BYTE,w,h));
            void * dataMap = mipmaps[nmipmap]->map();
            ILint size = ilGetInteger(IL_IMAGE_SIZE_OF_DATA);
            //std::cout<<size<<std::endl;
            memcpy(dataMap,data,size);
            mipmaps[nmipmap]->unmap();
            mipmaps[nmipmap]->validate();
            nmipmap++;
            ilBindImage(image);
            iluBuildMipmaps();
        }
        res->setMipLevelCount(nmipmap);
        for(int i=0; i<nmipmap; i++){
            res->setBuffer(0,i,mipmaps[i]);
        }
        res->validate();
    }
    else{
        res->destroy();
        std::cout<<"Error reading texture: "<<name<<std::endl;
        return NULL;
    }
    ilBindImage(0);
    ilDeleteImage(image);
    return res;
}

inline std::map<std::string,TextureSampler> loadTextures(const aiScene *s)
{
    ilInit();
    std::map<std::string,TextureSampler> textureNameMap;
    std::map<std::string,int> names;
    int countTex=0;
    for(unsigned int m=0; m< s->mNumMaterials; ++m)
    {
        int texIndex=0;
        aiString path;
        aiReturn texFound = s->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE,texIndex,&path);
        while(texFound==AI_SUCCESS)
        {
            texIndex++;
            texFound = s->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE,texIndex,&path);
            countTex++;
            names[path.data]=0;
        }
    }
    ILuint imageIds[countTex];
    ilGenImages(countTex,imageIds);
    std::map<std::string,int>::iterator itr= names.begin();
    for( int i=0; itr!=names.end(); itr++,i++)
    {
        std::string filename=itr->first;
        textureNameMap[filename]=newTexture(filename);
        std::cout<<"Successfully loaded texture: "<<filename<<std::endl;
    }
    return textureNameMap;
}


inline std::vector<Material> loadMaterials(const aiScene *s, std::map<std::string,TextureSampler> texMap, std::map<std::string,int> &matNameToIndex)
{
    std::vector<Material> res;
    Program closest_hit_radiance = renderer->createProgramFromPTXFile(ptx_p,"closest_hit_radiance");
    Program any_hit_shadow = renderer->createProgramFromPTXFile(ptx_p,"any_hit_shadow");
    Program any_hit_radiance = renderer->createProgramFromPTXFile(ptx_p,"any_hit_radiance");

    Buffer noBuffer = renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_BYTE4,1,1);
    Buffer noBufferBump = renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_BYTE,1,1);

    TextureSampler noTex=renderer->createTextureSampler();
    noTex->setWrapMode(0,RT_WRAP_CLAMP_TO_EDGE);
    noTex->setWrapMode(1,RT_WRAP_CLAMP_TO_EDGE);
    noTex->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    noTex->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    noTex->setFilteringModes(RT_FILTER_LINEAR,RT_FILTER_LINEAR,RT_FILTER_NONE);
    noTex->setMipLevelCount(1);
    noTex->setMaxAnisotropy(ANISOTROPY);
    noTex->setArraySize(1);
    noTex->setBuffer(0,0,noBuffer);

    TextureSampler noTexBump=renderer->createTextureSampler();
    noTexBump->setWrapMode(0,RT_WRAP_CLAMP_TO_EDGE);
    noTexBump->setWrapMode(1,RT_WRAP_CLAMP_TO_EDGE);
    noTexBump->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    noTexBump->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    noTexBump->setFilteringModes(RT_FILTER_LINEAR,RT_FILTER_LINEAR,RT_FILTER_NONE);
    noTexBump->setMipLevelCount(1);
    noTexBump->setMaxAnisotropy(ANISOTROPY);
    noTexBump->setArraySize(1);
    noTexBump->setBuffer(0,0,noBufferBump);

    for(unsigned int m=0; m<s->mNumMaterials; m++)
    {
        Material optix_mat=renderer->createMaterial();
        aiMaterial *mat = s->mMaterials[m];
        aiString texPath;
        aiString mat_name;
        aiGetMaterialString(mat,AI_MATKEY_NAME,&mat_name);
        std::cout<<"Loading material: "<<mat_name.data<<std::endl;
        if(AI_SUCCESS==mat->GetTexture(aiTextureType_DIFFUSE,0,&texPath))
        {
            std::cout<<"Texture: "<<texPath.data<<std::endl;
            optix_mat["tex0"]->setTextureSampler(texMap[texPath.data]);
            optix_mat["texCount"]->setInt(1);
        }
        else
        {
            optix_mat["tex0"]->setTextureSampler(noTex);
            optix_mat["texCount"]->setInt(0);
        }

        if(AI_SUCCESS==mat->GetTexture(aiTextureType_HEIGHT,0,&texPath))
        {
            std::cout<<"Bump: "<<texPath.data<<std::endl;
            optix_mat["bump"]->setTextureSampler(newTextureBump(texPath.data));
            optix_mat["bumpCount"]->setInt(1);
        }
        else
        {
            optix_mat["bump"]->setTextureSampler(noTex);
            optix_mat["bumpCount"]->setInt(0);
        }

        aiColor4D diffuse;
        if(AI_SUCCESS==aiGetMaterialColor(mat,AI_MATKEY_COLOR_DIFFUSE,&diffuse))
        {
            float4 temp;
            temp.x=diffuse.r;
            temp.y=diffuse.g;
            temp.z=diffuse.b;
            temp.w=diffuse.a;
            optix_mat["diffuse"]->setFloat(temp);
            std::cout<<"Diffuse: "<<temp.x<<' '<<temp.y<<' '<<temp.z<<' '<<temp.w<<std::endl;
        }
        aiColor4D spec;
        if(AI_SUCCESS==aiGetMaterialColor(mat,AI_MATKEY_COLOR_SPECULAR,&spec))
        {
            float4 temp;
            temp.x=spec.r;
            temp.y=spec.g;
            temp.z=spec.b;
            temp.w=spec.a;
            //float spec_inten=0.f;
            //aiGetMaterialFloat(mat,AI_MATKEY_SHININESS_STRENGTH,&spec_inten);
            //temp*=spec_inten;
            optix_mat["specular"]->setFloat(temp);
            std::cout<<"Specular: "<<temp.x<<' '<<temp.y<<' '<<temp.z<<' '<<temp.w<<std::endl;
        }
        float shininess;
        if(AI_SUCCESS==aiGetMaterialFloat(mat,AI_MATKEY_SHININESS,&shininess))
        {
            optix_mat["shininess"]->setFloat(shininess);
            std::cout<<"Shininess: "<<shininess<<std::endl;
        }
        float ior;
        aiGetMaterialFloat(mat,AI_MATKEY_REFRACTI,&ior);
        std::cout<<"Index of refraction: "<<ior<<std::endl;

        std::cout<<"Loaded material: "<<mat_name.data<<std::endl;

        optix_mat->setClosestHitProgram(Phong,closest_hit_radiance);
        optix_mat->setAnyHitProgram(Shadow,any_hit_shadow);
        optix_mat->setAnyHitProgram(Phong,any_hit_radiance);
        res.push_back(optix_mat);
        matNameToIndex[mat_name.data]=m;
        optix_mat->validate();
    }

    return res;
}

Acceleration newAccelerator(){
    Acceleration acc=renderer->createAcceleration("Bvh","Bvh");
    return acc;
}

Acceleration newAcceleratorGeom(){
    //Acceleration acc=renderer->createAcceleration("TriangleKdTree","KdTree");
    Acceleration acc=renderer->createAcceleration("Sbvh","Bvh");
    acc->setProperty("vertex_buffer_name","vertex_buffer");
    acc->setProperty("vertex_buffer_stride","0");
    acc->setProperty("index_buffer_name","index_buffer");
    acc->setProperty("index_buffer_stride","0");
    return acc;
}

GeometryGroup loadGeometryGroup(aiNode* node, GeometryInstance meshes[])
{
    GeometryGroup geom_g=renderer->createGeometryGroup();
    geom_g->setChildCount(node->mNumMeshes);
    geom_g->setAcceleration(newAcceleratorGeom());
    for(unsigned int m=0; m<node->mNumMeshes; m++)
    {
        GeometryInstance instance=meshes[node->mMeshes[m]];
        geom_g->setChild(m,instance);
    }
    geom_g->validate();
    return geom_g;
}

Transform loadNode(aiNode* node, GeometryInstance meshes[])
{
    Group child=renderer->createGroup();
    aiMatrix4x4 trans = node->mTransformation;
    Transform optix_trans=renderer->createTransform();
    float mat_arr[16]={trans.a1, trans.b1, trans.c1, trans.d1,
                       trans.a2, trans.b2, trans.c2, trans.d2,
                       trans.a3, trans.b3, trans.c3, trans.d3,
                       trans.a4, trans.b4, trans.c4, trans.d4};

    /*std::cout<<mat_arr[0]<<' '<<mat_arr[1]<<' '<<mat_arr[2]<<' '<<mat_arr[3]<<std::endl
             <<mat_arr[4]<<' '<<mat_arr[5]<<' '<<mat_arr[6]<<' '<<mat_arr[7]<<std::endl
             <<mat_arr[8]<<' '<<mat_arr[9]<<' '<<mat_arr[10]<<' '<<mat_arr[11]<<std::endl
             <<mat_arr[12]<<' '<<mat_arr[13]<<' '<<mat_arr[14]<<' '<<mat_arr[15]<<std::endl<<std::endl;*/

    Matrix4x4 mat(mat_arr);
    Matrix4x4 mat_inv=mat.inverse();
    optix_trans->setMatrix(true,mat.getData(),mat_inv.getData());
    optix_trans->setChild(child);
    GeometryGroup geom_g=loadGeometryGroup(node,meshes);

    if(node->mNumChildren>0){
        child->setAcceleration(newAccelerator());
    }
    else{
        child->setAcceleration(renderer->createAcceleration("NoAccel","NoAccel"));
    }
    child->setChildCount(1+node->mNumChildren);
    for(unsigned int m=0; m<node->mNumChildren;m++)
    {
        Transform t=loadNode(node->mChildren[m],meshes);
        child->setChild(m,t);
    }
    child->setChild(node->mNumChildren,geom_g);
    child->validate();
    optix_trans->validate();
    return optix_trans;
}


inline Group loadGeometry(const aiScene * s, std::vector<Material> materialVec)
{
    Buffer noTexCoord=renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT2,1);
    Program bounding_box = renderer->createProgramFromPTXFile(ptx_p,"boundingBoxMesh");
    Program intersect = renderer->createProgramFromPTXFile(ptx_p,"intersectMesh");
    GeometryInstance meshes[s->mNumMeshes];
    for(unsigned int m=0; m<s->mNumMeshes; m++)
    {
        std::cout<<"Loading mesh: "<<m<<std::endl;
        //Initialize Geometry
        std::cout<<"Initializing"<<std::endl;
        aiMesh * mesh=s->mMeshes[m];
        Geometry optix_mesh=renderer->createGeometry();
        optix_mesh->setPrimitiveCount(mesh->mNumFaces);
        //copy indices buffer to optix
        std::cout<<"Loading Indices"<<std::endl;
        optix_mesh->setPrimitiveCount(mesh->mNumFaces);
        Buffer index_buffer = renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_INT3,mesh->mNumFaces);
        int *temp_index=static_cast<int*>(index_buffer->map());
        for(unsigned int i=0; i<mesh->mNumFaces; i++)
        {
            temp_index[3*i]=mesh->mFaces[i].mIndices[0];
            temp_index[3*i+1]=mesh->mFaces[i].mIndices[1];
            temp_index[3*i+2]=mesh->mFaces[i].mIndices[2];
        }
        index_buffer->unmap();
        index_buffer->validate();
        //copy vertices buffer
        std::cout<<"Loading Vertices"<<std::endl;
        Buffer vertex_buffer=renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT3,mesh->mNumVertices);
        void * temp_pos=vertex_buffer->map();
        memcpy(temp_pos,mesh->mVertices,mesh->mNumVertices*3*sizeof(float));
        vertex_buffer->unmap();
        vertex_buffer->validate();
        //copy normals
        std::cout<<"Loading Normals"<<std::endl;
        Buffer normal_buffer=renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT3,mesh->mNumVertices);
        void * temp_norm=normal_buffer->map();
        memcpy(temp_norm,mesh->mNormals,mesh->mNumVertices*3*sizeof(float));
        normal_buffer->unmap();
        normal_buffer->validate();
        //copy tangents

        Buffer tangent_buffer;
        Buffer bitangent_buffer;
        if(mesh->HasTangentsAndBitangents()){
            std::cout<<"Loading Tangents"<<std::endl;
            tangent_buffer=renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT3,mesh->mNumVertices);
            bitangent_buffer=renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT3,mesh->mNumVertices);

            void * temp_tan=tangent_buffer->map();
            memcpy(temp_tan,mesh->mTangents,mesh->mNumVertices*3*sizeof(float));
            tangent_buffer->unmap();

            void * temp_bitan=bitangent_buffer->map();
            memcpy(temp_bitan,mesh->mBitangents,mesh->mNumVertices*3*sizeof(float));
            bitangent_buffer->unmap();
        }
        //copy tex coordinates
        Buffer texCoord_buffer;
        if(mesh->HasTextureCoords(0))
        {
            std::cout<<"Loading TexCoord"<<std::endl;
            texCoord_buffer=renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT2,mesh->mNumVertices);
            float * temp_texCoord=static_cast<float*>(texCoord_buffer->map());
            for(unsigned int i=0; i<mesh->mNumVertices; i++)
            {
                temp_texCoord[2*i]=mesh->mTextureCoords[0][i].x;
                temp_texCoord[2*i+1]=mesh->mTextureCoords[0][i].y;
                //std::cout<<"Loading TexCoord "<< mesh->mTextureCoords[0][i].x<<' '<<mesh->mTextureCoords[0][i].x <<std::endl;
            }
            texCoord_buffer->unmap();
        }
        //set atributes
        std::cout<<"Setting Attributes"<<std::endl;
        optix_mesh["vertex_buffer"]->set(vertex_buffer);
        optix_mesh["index_buffer"]->set(index_buffer);
        optix_mesh["normal_buffer"]->set(normal_buffer);
        if(mesh->HasTextureCoords(0)) {
            optix_mesh["texCoord_buffer"]->set(texCoord_buffer);
            optix_mesh["hasTexCoord"]->setInt(1);
        }
        else{
            optix_mesh["texCoord_buffer"]->set(noTexCoord);
            optix_mesh["hasTexCoord"]->setInt(0);
        }
        if(mesh->HasTangentsAndBitangents()) {
            optix_mesh["tangent_buffer"]->set(tangent_buffer);
            optix_mesh["bitangent_buffer"]->set(bitangent_buffer);
            optix_mesh["hasTangents"]->setInt(1);
        }
        else{
            optix_mesh["tangent_buffer"]->set(renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT3,1));
            optix_mesh["bitangent_buffer"]->set(renderer->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT3,1));
            optix_mesh["hasTangents"]->setInt(0);

        }
        //set optix programs
        std::cout<<"Setting Programs"<<std::endl;
        optix_mesh->setBoundingBoxProgram(bounding_box);
        optix_mesh->setIntersectionProgram(intersect);
        //create geometry instance
        std::cout<<"Instanctiating"<<std::endl;
        GeometryInstance instance=renderer->createGeometryInstance();

        instance->setGeometry(optix_mesh);
        instance->setMaterialCount(1);
        instance->setMaterial(0,materialVec[mesh->mMaterialIndex]);


        //std::cout<<matName<<std::endl;
        meshes[m]=instance;
        std::cout<<"Loaded mesh: "<<m<<std::endl<<std::endl;
        optix_mesh->validate();
        instance->validate();
    }
    Transform t=loadNode(s->mRootNode,meshes);
    Group top = renderer->createGroup();
    top->setChildCount(1);
    top->setAcceleration(renderer->createAcceleration("NoAccel","NoAccel"));
    top->setChild(0,t);
    top->validate();
    return top;
}



void inline initContext()
{
    //create context
    renderer=Context::create();
    renderer->setRayTypeCount(RAY_TYPE_COUNT);
    renderer["Phong"]->setInt(Phong);
    renderer["Shadow"]->setInt(Shadow);

    const aiScene * scene = loadScene(scene_p+scene_name);
    std::map<std::string,TextureSampler> texMap=loadTextures(scene);
    std::map<std::string,int> matNameToIndex;
    std::vector<Material> materials=loadMaterials(scene,texMap,matNameToIndex);
    Group top=loadGeometry(scene,materials);
    renderer["top_object"]->set(top);

    Program miss_radiance = renderer->createProgramFromPTXFile(ptx_p,"miss_radiance");
    Program miss_shadow = renderer->createProgramFromPTXFile(ptx_p,"miss_shadow");

    Program entryPoint=renderer->createProgramFromPTXFile(ptx_p,"pinhole_camera");

    Program entryPoint_ms=renderer->createProgramFromPTXFile(ptx_p,"pinhole_camera_ms");

    Program exept=renderer->createProgramFromPTXFile(ptx_p,"exception");


    renderer->setEntryPointCount(ENTRY_COUNT);
    renderer->setRayGenerationProgram(ENTRY_PINHOLE,entryPoint);
    renderer->setRayGenerationProgram(ENTRY_PINHOLE_MS,entryPoint_ms);

    for(int i=0; i<ENTRY_COUNT; i++){
        renderer->setExceptionProgram(i,exept);
    }

    renderer->setExceptionEnabled(RT_EXCEPTION_ALL,true);

    renderer->setStackSize(1500);

    renderer->setMissProgram(Phong,miss_radiance);
    renderer->setMissProgram(Shadow,miss_shadow);

    out=genOutputBuffer();
    renderer["output0"]->set(out);

    float3 V=normalize(cross(up,-lookDir));
    float3 U=cross(-lookDir,V);

    renderer["eye"]->setFloat(eye);
    renderer["U"]->setFloat(U);
    renderer["V"]->setFloat(V);
    renderer["W"]->setFloat(lookDir);
    renderer["fov"]->setFloat(1.f);

    renderer["lightDir"]->setFloat(normalize(make_float3(-0.5f,-5.f,-1.f)));

    TextureSampler sky = newTexture("../skydome.png");

    renderer["sky"]->set(sky);

    renderer->validate();
}

void keyboard(unsigned char key, int x, int y){

    float3 V=normalize(cross(up,-lookDir));
    float3 U=cross(-lookDir,V);

    switch(key){
    case 'w':
        eye+=STEP*lookDir;
        break;
    case 's':
        eye-=STEP*lookDir;
        break;

    case 'i':
        lookDir=normalize(lookDir+ANG_STEP*U);
        break;
    case 'k':
        lookDir=normalize(lookDir-ANG_STEP*U);
        break;

    case 'l':
        lookDir=normalize(lookDir+ANG_STEP*V);
        break;
    case 'j':
        lookDir=normalize(lookDir-ANG_STEP*V);
        break;
    }

    V=normalize(cross(up,-lookDir));
    U=cross(-lookDir,V);

    renderer["eye"]->setFloat(eye);
    renderer["U"]->setFloat(U);
    renderer["V"]->setFloat(V);
    renderer["W"]->setFloat(lookDir);
}


int main(int argc, char ** argv)
{
    //init glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE);
    glutInitWindowPosition(0,0);
    glutInitWindowSize(width,height);
    glutCreateWindow("OptiX Ray Tracing Engine");
    //callbacks
    glutReshapeFunc(reshape);
    glutDisplayFunc(renderScene);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(renderScene);
    //init glew
    glewInit();
    //setup optix
    ilInit();
    initContext();
    //main loop
    glutMainLoop();
    return 0;
}
