#ifndef OPTIXRENDERER_H
#define OPTIXRENDERER_H

#include <map>
#include <optix_world.h>
#include <assimp/scene.h>



class OptixRenderer
{
    public:
        OptixRenderer(std::string path, std::string file);
        virtual ~OptixRenderer();

        void setRayTypeCount(int n);
        void setOutputSize(int w, int h);

        void setEntryProgram(std::string file, std::string program);
        void setExceptionProgram(std::string file, std::string program);
        void setMissProgram(int ray_type, std::string file, std::string program);

        void setIntersectionProgram(std::string file, std::string program);
        void setBoundingBoxProgram(std::string file, std::string program);

        void setDefaultClosestHitProgram(int ray_type, std::string file, std::string program);
        void setDefaultAnyHitProgram(int ray_type, std::string file, std::string program);

        void setMaterialClosestHitProgram(std::string mat_name, int ray_type, std::string file, std::string program);
        void setMaterialAnyHitProgram(std::string mat_name, int ray_type, std::string file, std::string program);

        void init();

        inline void run();

        void* mapOutputBuffer();
        void unmapOutputBuffer();

        optix::Variable variable(const std::string &name);

    protected:
    private:
        std::string scene_path, scene_file;

        optix::TextureSampler createTextureRGBA(std::string file);
        optix::TextureSampler createTextureLum(std::string file);


        void loadMaterials();
        void loadGeometry();
        void loadSceneGraph();

        optix::Acceleration createAccelerationMeshes();
        optix::Acceleration createAccelerationGroups();

        optix::Transform loadNode(aiNode * node);
        optix::GeometryGroup loadGeometryGroup(aiNode * node);

        optix::Context context;
        optix::Buffer output;
        const aiScene *scene;
        std::map<std::string, optix::Material> materials;
        std::vector<optix::GeometryInstance> meshes;
        optix::Transform top;

        optix::Program bounding_box;
        optix::Program intersect;

        int width, height;
};

#endif // OPTIXRENDERER_H
