#include "OptixRenderer.h"

#include <assimp/cimport.h>
#include <assimp/cexport.h>
#include <assimp/postprocess.h>
#include <assimp/material.h>

#include <IL/il.h>
#include <IL/ilu.h>


#define ANISOTROPY 1.f
#define MIPMAPS 1

using namespace std;
using namespace optix;

OptixRenderer::OptixRenderer(string path, string file) : materials(), meshes()
{
    //ctor
    //creating context
    context=Context::create();
    //loading scene
    scene_path=path;
    scene_file=file;
    scene=aiImportFile((path+file).c_str(), aiProcessPreset_TargetRealtime_MaxQuality | aiProcess_OptimizeGraph);

    loadMaterials();
    loadGeometry();
    loadSceneGraph();

}

OptixRenderer::~OptixRenderer()
{
    //dtor
    context->destroy();
}


void OptixRenderer::loadMaterials(){
    int nmat=scene->mNumMaterials;
    for(int i=0; i<nmat; i++){

        aiMaterial * mat = scene->mMaterials[i];
        Material optix_mat = context->createMaterial();

        aiString mat_name;
        aiGetMaterialString(mat, AI_MATKEY_NAME, &mat_name);

        aiColor4D diffuse;
        aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diffuse);
        optix_mat["Kd"]->setFloat(diffuse.r, diffuse.b, diffuse.g, diffuse.a);

        aiColor4D specular;
        aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &specular);
        optix_mat["Ks"]->setFloat(specular.r, specular.b, specular.g, specular.a);

        float shininess;
        aiGetMaterialFloat(mat, AI_MATKEY_SHININESS, &shininess);
        optix_mat["Ns"]->setFloat(shininess);

        aiString diffTexPath;
        if(AI_SUCCESS==mat->GetTexture(aiTextureType_DIFFUSE, 0, &diffTexPath)){
            TextureSampler diffTex = createTextureRGBA(scene_path+string(diffTexPath.data));
            optix_mat["map_Kd"]->setTextureSampler(diffTex);

        }
        else{
            optix_mat["map_Kd"]->setTextureSampler(createTextureRGBA(""));
        }

        aiString specTexPath;
        if(AI_SUCCESS==mat->GetTexture(aiTextureType_SPECULAR, 0, &specTexPath)){
            TextureSampler specTex = createTextureRGBA(scene_path+string(specTexPath.data));
            optix_mat["map_Ks"]->setTextureSampler(specTex);
        }
        else{
            optix_mat["map_Ks"]->setTextureSampler(createTextureRGBA(""));
        }

        aiString bumpTexPath;
        if(AI_SUCCESS==mat->GetTexture(aiTextureType_HEIGHT, 0, &bumpTexPath)){
            TextureSampler bumpTex = createTextureLum(scene_path+string(bumpTexPath.data));
            optix_mat["map_bump"]->setTextureSampler(bumpTex);
        }
        else{
            optix_mat["map_bump"]->setTextureSampler(createTextureLum(""));
        }

        optix_mat->validate();
        materials[mat_name.data]=optix_mat;
    }
}

void OptixRenderer::loadGeometry(){

    int nmeshes = scene->mNumMeshes;

    for(int i=0; i<nmeshes; i++){

        aiMesh * mesh = scene->mMeshes[i];
        int nprimitive = mesh->mNumFaces;
        int nvertex = mesh->mNumVertices;

        Geometry optix_mesh = context->createGeometry();
        optix_mesh->setPrimitiveCount(nprimitive);

        Buffer index_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_INT3,nprimitive);
        int *tmp_index = static_cast<int * >(index_buffer->map());
        for(int p=0; p<nprimitive; p++){
            memcpy(tmp_index+3*p,mesh->mFaces[p].mIndices,3*sizeof(int));
        }
        index_buffer->unmap();
        index_buffer->validate();
        optix_mesh["index_buffer"]->set(index_buffer);

        Buffer vertex_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, nvertex);
        void *tmp_vertex = vertex_buffer->map();
        memcpy(tmp_vertex, mesh->mVertices, nvertex*3*sizeof(float));
        vertex_buffer->unmap();
        vertex_buffer->validate();
        optix_mesh["vertex_buffer"]->set(vertex_buffer);

        Buffer normal_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, nvertex);
        void *tmp_normal = normal_buffer->map();
        memcpy(tmp_normal, mesh->mNormals, nvertex*3*sizeof(float));
        normal_buffer->unmap();
        normal_buffer->validate();
        optix_mesh["normal_buffer"]->set(normal_buffer);

        Buffer tangent_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT3, nvertex);
        void *tmp_tangent = tangent_buffer->map();
        memcpy(tmp_tangent, mesh->mTangents, nvertex*3*sizeof(float));
        tangent_buffer->unmap();
        tangent_buffer->validate();
        optix_mesh["tangent_buffer"]->set(tangent_buffer);

        Buffer bitangent_buffer = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT3, nvertex);
        void *tmp_bitangent = bitangent_buffer->map();
        memcpy(tmp_bitangent, mesh->mBitangents, nvertex*3*sizeof(float));
        bitangent_buffer->unmap();
        bitangent_buffer->validate();
        optix_mesh["bitangent_buffer"]->set(bitangent_buffer);

        if(mesh->HasTextureCoords(0)){
            Buffer texCoord_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, nvertex);
            float *tmp_texCoord = static_cast<float * >(texCoord_buffer->map());
            for(int tc=0; tc<nvertex; tc++){
                tmp_texCoord[2*tc] = mesh->mTextureCoords[0][tc].x;
                tmp_texCoord[2*tc+1] = mesh->mTextureCoords[0][tc].y;
            }
            texCoord_buffer->unmap();
            texCoord_buffer->validate();
            optix_mesh["texCoord_buffer"]->set(texCoord_buffer);
            optix_mesh["hasTexCoord"]->setInt(1);
        }
        else{
            optix_mesh["texCoord_buffer"]->set(context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_FLOAT2, 1));
            optix_mesh["hasTexCoord"]->setInt(0);
        }

        optix_mesh->validate();

        GeometryInstance instance = context->createGeometryInstance();
        instance->setGeometry(optix_mesh);
        instance->setMaterialCount(1);

        aiString mat_name;
        aiGetMaterialString(scene->mMaterials[mesh->mMaterialIndex], AI_MATKEY_NAME, &mat_name);
        instance->setMaterial(0, materials[mat_name.data]);

        instance->validate();

        meshes.push_back(instance);
    }
}

void OptixRenderer::loadSceneGraph(){

}

TextureSampler OptixRenderer::createTextureRGBA(string file){

    ILuint image=iluGenImage();
    ilBindImage(image);
    ilEnable(IL_ORIGIN_SET);
    ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
    iluBuildMipmaps();

    TextureSampler res=context->createTextureSampler();;
    res->setArraySize(1);
    res->setWrapMode(0,RT_WRAP_REPEAT);
    res->setWrapMode(1,RT_WRAP_REPEAT);
    res->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    res->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    res->setMaxAnisotropy(ANISOTROPY);
    res->setFilteringModes(RT_FILTER_LINEAR,RT_FILTER_LINEAR,RT_FILTER_NONE);

    ILboolean success=ilLoadImage((ILstring) file.c_str());
    if(success){
        ilConvertImage(IL_RGBA,IL_UNSIGNED_BYTE);
        std::vector<Buffer> mipmaps;
        int nmipmap=0;
        while(ilActiveMipmap(nmipmap)&&nmipmap<MIPMAPS){
            int w=ilGetInteger(IL_IMAGE_WIDTH);
            int h=ilGetInteger(IL_IMAGE_HEIGHT);

            void * data= (void*)ilGetData();
            mipmaps.push_back(context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_UNSIGNED_BYTE4,w,h));
            void * dataMap = mipmaps[nmipmap]->map();
            ILint size = ilGetInteger(IL_IMAGE_SIZE_OF_DATA);

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

    }
    else{
        res->setMipLevelCount(1u);
        Buffer white = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_UNSIGNED_BYTE4,1,1);
        unsigned char * bytes = static_cast<unsigned char *>(white->map());
        bytes[0]=255;
        bytes[1]=255;
        bytes[2]=255;
        bytes[3]=255;
        white->unmap();
        white->validate();
        res->setBuffer(0,0,white);
    }

    res->validate();

    ilBindImage(0);
    ilDeleteImage(image);
    return res;
}

TextureSampler OptixRenderer::createTextureLum(std::string file)
{
    ILuint image=iluGenImage();
    ilBindImage(image);
    ilEnable(IL_ORIGIN_SET);
    ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
    iluBuildMipmaps();

    TextureSampler res=context->createTextureSampler();
    res->setArraySize(1);

    res->setWrapMode(0,RT_WRAP_REPEAT);
    res->setWrapMode(1,RT_WRAP_REPEAT);
    res->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    res->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    res->setMaxAnisotropy(ANISOTROPY);
    res->setFilteringModes(RT_FILTER_LINEAR,RT_FILTER_LINEAR,RT_FILTER_NONE);

    ILboolean success=ilLoadImage((ILstring) file.c_str());
    if(success){
        ilConvertImage(IL_LUMINANCE,IL_UNSIGNED_BYTE);
        std::vector<Buffer> mipmaps;
        int nmipmap=0;
        while(ilActiveMipmap(nmipmap)&&nmipmap<MIPMAPS){
            int w=ilGetInteger(IL_IMAGE_WIDTH);
            int h=ilGetInteger(IL_IMAGE_HEIGHT);
            //std::cout<<w<<'x'<<h<<std::endl;
            void * data= (void*)ilGetData();
            mipmaps.push_back(context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_UNSIGNED_BYTE,w,h));
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
    }
    else{
        res->setMipLevelCount(1u);
        Buffer white = context->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_UNSIGNED_BYTE,1,1);
        unsigned char * bytes = static_cast<unsigned char *>(white->map());
        bytes[0]=255;
        white->unmap();
        white->validate();
        res->setBuffer(0,0,white);
    }
    ilBindImage(0);
    ilDeleteImage(image);
    res->validate();
    return res;
}
