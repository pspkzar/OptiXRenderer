#include "OptixRenderer.h"

#include <assimp/cimport.h>
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
    }
}

void OptixRenderer::loadGeometry(){

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
        res->validate();
    }
    else{
        res->destroy();
        cout<<"Error reading texture "<<file<<endl;
        exit(0);
    }
    ilBindImage(0);
    ilDeleteImage(image);
    return res;
}
