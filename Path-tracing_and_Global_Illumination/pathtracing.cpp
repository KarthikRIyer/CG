#define M_PI 3.14159265358979323846
#include <stdio.h>
#include <random>
#include <vector>
#include <memory>
#include <stdlib.h>
#include <utility>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>
#include <math.h>

#include "geometry.h"

using namespace std;

static const float kInfinity = numeric_limits<float>::max();

static const float kEpsilon = 1e-8;

static const Vec3f kDefaultBackgroundColor = Vec3f(0.235294, 0.67451, 0.843137);

//static template <> const Matrix44f Matrix44f::kIdentity = Matrix44f();
static const Matrix44f kIdentity = Matrix44f();

inline float clamp(const float& lo, const float& hi, const float& v)
{

    return max(lo, min(hi, v));
}

inline float degToRad(const float& deg)
{

    return (deg * M_PI) / 180.0f;
}

inline Vec3f lerp(const Vec3f& a, const Vec3f& b, const float& interpolate)
{

    return (a * (1 - interpolate) + b * (interpolate));
}

struct Options {

    uint32_t width = 640;
    uint32_t height = 480;

    float fieldOfView = 90.0f;

    Vec3f backgroundColor = kDefaultBackgroundColor;

    Matrix44f cameraToWorld;

    float bias = 0.0001f;

    uint32_t maxDepth = 2;
};

enum MaterialType { kDiffuse };

class Object {

public:
    virtual ~Object() {}

    Object(const Matrix44f& oToW)
        : objectToWorld(oToW)
        , worldToObject(oToW.inverse())
    {
    }

    virtual bool intersect(const Vec3f&, const Vec3f&, float&, uint32_t&, Vec2f&) const = 0;

    virtual void getSurfaceProperties(const Vec3f&, const Vec3f&, const uint32_t&, const Vec2f&, Vec3f&, Vec2f&) const = 0;

    Matrix44f objectToWorld, worldToObject;

    MaterialType type = kDiffuse;

    Vec3f albedo = 0.18f;

    float Kd = 0.8f;
    float Ks = 0.2f;
    float n = 10;
};

bool solveQuadratic(float& a, float& b, float& c, float& x0, float& x1)
{

    float D = b * b - 4 * a * c;

    if (D < 0)
        return false;
    else if (D == 0) {
        x0 = x1 = (-b / (2.0f * a));
    }
    else {

        float rootD = (float)sqrt(D);

        x0 = (-b + rootD) / (2.0f * a);
        x1 = (-b - rootD) / (2.0f * a);
    }
    return true;
}

class Sphere : public Object {

public:
    float radius, radius2;
    Vec3f center;
    Sphere(const Matrix44f& o2w, const float& r)
        : Object(o2w)
        , radius(r)
        , radius2(r * r)
    {
        o2w.multVecMatrix(Vec3f(0), center);
    };

    bool intersect(const Vec3f& origin, const Vec3f& direction, float& tNear, uint32_t& triIndex, Vec2f& uv) const
    {

        float t0, t1;

        Vec3f v = origin - center;

        float a = direction.dotProduct(direction);
        float b = 2 * direction.dotProduct(v);
        float c = v.dotProduct(v) - radius2;
        if (!solveQuadratic(a, b, c, t0, t1)) {
            return false;
        }

        if (t0 > t1)
            swap(t0, t1);

        if (t0 < 0) {
            t0 = t1;
            if (t0 < 0)
                return false;
        }
        tNear = t0;

        return true;
    }

    void getSurfaceProperties(const Vec3f& hitPoint, const Vec3f& viewDirection, const uint32_t& trIndex, const Vec2f& uv, Vec3f& hitNormal, Vec2f& hitTextureCoordinates) const
    {

        hitNormal = hitPoint - center;
        hitNormal.normalize();

        hitTextureCoordinates.x = (1 + atan2(hitNormal.z, hitNormal.x) / M_PI) * 0.5f;
        hitTextureCoordinates.y = acosf(hitNormal.y) / M_PI;
    }
};

bool rayTriangleIntersect(const Vec3f& origin, const Vec3f& direction, const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, float& t, float& u, float& v)
{

    Vec3f v0v1 = v1 - v0;
    Vec3f v0v2 = v2 - v0;
    Vec3f pvec = direction.crossProduct(v0v2);
    float det = v0v1.dotProduct(pvec);

    if (fabs(det) < kEpsilon) {
        return false;
    }

    float invDet = 1 / det;

    Vec3f tvec = origin - v0;
    u = tvec.dotProduct(pvec) * invDet;
    if (u < 0 || u > 1) {
        return false;
    }

    Vec3f qvec = tvec.crossProduct(v0v1);
    v = direction.dotProduct(qvec) * invDet;
    if (v < 0 || u + v > 1)
        return false;

    t = v0v2.dotProduct(qvec) * invDet;

    return (t > 0) ? true : false;
}

class TriangleMesh : public Object {

public:
    TriangleMesh(
        const Matrix44f& o2w,
        const uint32_t nfaces,
        const std::unique_ptr<uint32_t[]>& faceIndex,
        const std::unique_ptr<uint32_t[]>& vertsIndex,
        const std::unique_ptr<Vec3f[]>& verts,
        std::unique_ptr<Vec3f[]>& normals,
        std::unique_ptr<Vec2f[]>& st)
        : Object(o2w)
        , numTris(0)
    {
        uint32_t k = 0, maxVertIndex = 0;
        // find out how many triangles we need to create for this mesh
        for (uint32_t i = 0; i < nfaces; ++i) {
            numTris += faceIndex[i] - 2;
            for (uint32_t j = 0; j < faceIndex[i]; ++j)
                if (vertsIndex[k + j] > maxVertIndex)
                    maxVertIndex = vertsIndex[k + j];
            k += faceIndex[i];
        }
        maxVertIndex += 1;

        // allocate memory to store the position of the mesh vertices
        P = std::unique_ptr<Vec3f[]>(new Vec3f[maxVertIndex]);
        for (uint32_t i = 0; i < maxVertIndex; ++i) {
            objectToWorld.multVecMatrix(verts[i], P[i]);
        }

        // allocate memory to store triangle indices
        trisIndex = std::unique_ptr<uint32_t[]>(new uint32_t[numTris * 3]);
        uint32_t l = 0;
        N = std::unique_ptr<Vec3f[]>(new Vec3f[numTris * 3]);
        sts = std::unique_ptr<Vec2f[]>(new Vec2f[numTris * 3]);
        Matrix44f transformNormals = worldToObject.transpose();
        // generate the triangle index array and set normals and st coordinates
        for (uint32_t i = 0, k = 0; i < nfaces; ++i) { // for each  face
            for (uint32_t j = 0; j < faceIndex[i] - 2; ++j) { // for each triangle in the face
                trisIndex[l] = vertsIndex[k];
                trisIndex[l + 1] = vertsIndex[k + j + 1];
                trisIndex[l + 2] = vertsIndex[k + j + 2];
                transformNormals.multDirMatrix(normals[k], N[l]);
                transformNormals.multDirMatrix(normals[k + j + 1], N[l + 1]);
                transformNormals.multDirMatrix(normals[k + j + 2], N[l + 2]);
                N[l].normalize();
                N[l + 1].normalize();
                N[l + 2].normalize();
                sts[l] = st[k];
                sts[l + 1] = st[k + j + 1];
                sts[l + 2] = st[k + j + 2];
                l += 3;
            }
            k += faceIndex[i];
        }
    }
    // Test if the ray interesects this triangle mesh
    bool intersect(const Vec3f& origin, const Vec3f& direction, float& tNear, uint32_t& triIndex, Vec2f& uv) const
    {

        uint32_t j = 0;

        bool isect = false;

        for (uint32_t i = 0; i < numTris; i++) {

            const Vec3f& v0 = P[trisIndex[j]];
            const Vec3f& v1 = P[trisIndex[j + 1]];
            const Vec3f& v2 = P[trisIndex[j + 2]];

            float t = kInfinity, u, v;

            if (rayTriangleIntersect(origin, direction, v0, v1, v2, t, u, v) && t < tNear) {

                tNear = t;
                uv.x = u;
                uv.y = v;
                triIndex = i;
                isect = true;
            }
            j += 3;
        }
        return isect;
    }

    void getSurfaceProperties(const Vec3f& hitPoint, const Vec3f& viewDirection, const uint32_t& triIndex, const Vec2f& uv, Vec3f& hitNormal, Vec2f& hitTextureCoordinates) const
    {

        if (smoothShading) {
            //vertex normal
            const Vec3f& n0 = N[triIndex * 3];
            const Vec3f& n1 = N[triIndex * 3 + 1];
            const Vec3f& n2 = N[triIndex * 3 + 2];
            hitNormal = (1 - uv.x - uv.y) * n0 + uv.x * n1 + uv.y * n2;
        }
        else {
            //face normal
            const Vec3f& v0 = N[triIndex * 3];
            const Vec3f& v1 = N[triIndex * 3 + 1];
            const Vec3f& v2 = N[triIndex * 3 + 2];
            hitNormal = (v1 - v0).crossProduct(v2 - v0);
        }

        hitNormal.normalize();

        // texture coordinates
        const Vec2f& st0 = sts[triIndex * 3];
        const Vec2f& st1 = sts[triIndex * 3 + 1];
        const Vec2f& st2 = sts[triIndex * 3 + 2];
        hitTextureCoordinates = (1 - uv.x - uv.y) * st0 + uv.x * st1 + uv.y * st2;
    }

    // member variables
    uint32_t numTris; // number of triangles
    std::unique_ptr<Vec3f[]> P; // triangles vertex position
    std::unique_ptr<uint32_t[]> trisIndex; // vertex index array
    std::unique_ptr<Vec3f[]> N; // triangles vertex normals
    std::unique_ptr<Vec2f[]> sts; // triangles texture coordinates
    bool smoothShading = true; // smooth shading by default
};

TriangleMesh* loadPolyMeshFromFile(const char* file, const Matrix44f& o2w)
{
    std::ifstream ifs;
    try {
        ifs.open(file);
        if (ifs.fail())
            throw;
        std::stringstream ss;
        ss << ifs.rdbuf();
        uint32_t numFaces;
        ss >> numFaces;
        std::unique_ptr<uint32_t[]> faceIndex(new uint32_t[numFaces]);
        uint32_t vertsIndexArraySize = 0;
        // reading face index array
        for (uint32_t i = 0; i < numFaces; ++i) {
            ss >> faceIndex[i];
            vertsIndexArraySize += faceIndex[i];
        }
        std::unique_ptr<uint32_t[]> vertsIndex(new uint32_t[vertsIndexArraySize]);
        uint32_t vertsArraySize = 0;
        // reading vertex index array
        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
            ss >> vertsIndex[i];
            if (vertsIndex[i] > vertsArraySize)
                vertsArraySize = vertsIndex[i];
        }
        vertsArraySize += 1;
        // reading vertices
        std::unique_ptr<Vec3f[]> verts(new Vec3f[vertsArraySize]);
        for (uint32_t i = 0; i < vertsArraySize; ++i) {
            ss >> verts[i].x >> verts[i].y >> verts[i].z;
        }
        // reading normals
        std::unique_ptr<Vec3f[]> normals(new Vec3f[vertsIndexArraySize]);
        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
            ss >> normals[i].x >> normals[i].y >> normals[i].z;
        }
        // reading st coordinates
        std::unique_ptr<Vec2f[]> st(new Vec2f[vertsIndexArraySize]);
        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
            ss >> st[i].x >> st[i].y;
        }

        return new TriangleMesh(o2w, numFaces, faceIndex, vertsIndex, verts, normals, st);
    }
    catch (...) {
        ifs.close();
    }
    ifs.close();

    return nullptr;
}

class Light {

public:
    Vec3f color;
    float intensity;
    Matrix44f lightToWorld;
    Light(const Matrix44f& l2w, const Vec3f& c = 1, const float& i = 1)
        : lightToWorld(l2w)
        , color(c)
        , intensity(i)
    {
    }
    virtual ~Light() {}
    virtual void illuminate(const Vec3f& P, Vec3f&, Vec3f&, float&) const = 0;
};

class DistantLight : public Light {

    Vec3f dir;

public:
    DistantLight(const Matrix44f& l2w, const Vec3f& c = 1, const float& i = 1)
        : Light(l2w, c, i)
    {

        l2w.multDirMatrix(Vec3f(0, 0, -1), dir);
        dir.normalize();
    }
    void illuminate(const Vec3f& P, Vec3f& lightDir, Vec3f& lightIntensity, float& distance) const
    {
        lightDir = dir;
        lightIntensity = color * intensity;
        distance = kInfinity;
    }
};

class PointLight : public Light {

    Vec3f pos;

public:
    PointLight(const Matrix44f& l2w, const Vec3f& c = 1, const float& i = 1)
        : Light(l2w, c, i)
    {

        l2w.multVecMatrix(Vec3f(0), pos);
    }
    //P is the shaded point
    void illuminate(const Vec3f& P, Vec3f& lightDir, Vec3f& lightIntensity, float& distance) const
    {

        lightDir = (P - pos);

        float r2 = lightDir.norm();

        distance = sqrt(r2);

        lightDir.x /= distance;
        lightDir.y /= distance;
        lightDir.z /= distance;

        lightIntensity = color * intensity / (float)(4.0f * M_PI * r2);
    }
};

enum RayType { kPrimaryRay,
    kShadowRay };

struct IntersectInfo {

    const Object* hitObject = nullptr;

    float tNear = kInfinity;

    Vec2f uv;

    uint32_t index = 0;
};

bool trace(const Vec3f& origin, const Vec3f& direction, const vector<unique_ptr<Object> >& objects, IntersectInfo& isect, RayType rayType = kPrimaryRay)
{

    isect.hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); k++) {

        float tNear = kInfinity;
        uint32_t index = 0;
        Vec2f uv;
        if (objects[k]->intersect(origin, direction, tNear, index, uv) && tNear < isect.tNear) {

            isect.hitObject = objects[k].get();
            isect.tNear = tNear;
            isect.index = index;
            isect.uv = uv;
        }
    }

    return (isect.hitObject != nullptr);
}

void createCoordinateSystem(const Vec3f& N, Vec3f& Nt, Vec3f& Nb)
{

    if (fabs(N.x) > fabs(N.y)) {

        Nt = Vec3f(N.z, 0, -N.x) / sqrtf(N.x * N.x + N.z * N.z);
    }
    else {

        Nt = Vec3f(0, -N.z, N.y) / sqrtf(N.y * N.y + N.z * N.z);
    }
    Nb = N.crossProduct(Nt);
}

Vec3f uniformSampleHemisphere(const float& r1, const float& r2)
{

    float sinTheta = (float)sqrt(1 - r1 * r1);
    float phi = 2 * M_PI * r2;
    float x = sinTheta * cosf(phi);
    float z = sinTheta * sinf(phi);
    return Vec3f(x, r1, z);
}

std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

Vec3f castRay(const Vec3f& origin, const Vec3f& direction, const vector<unique_ptr<Object> >& objects, const vector<unique_ptr<Light> >& lights, const Options& options, const uint32_t& depth = 0)
{

    if (depth > options.maxDepth) {
        return 0;
    }

    Vec3f hitColor = 0;

    IntersectInfo isect;

    if (trace(origin, direction, objects, isect)) {
        Vec3f hitPoint = origin + direction * isect.tNear;
        Vec3f hitNormal;
        Vec2f hitTextureCoordinates;
        isect.hitObject->getSurfaceProperties(hitPoint, direction, isect.index, isect.uv, hitNormal, hitTextureCoordinates);

        switch (isect.hitObject->type) {

        case kDiffuse: {
            //directLighting
            Vec3f directLighting = Vec3f(0);
            for (uint32_t i = 0; i < lights.size(); i++) {

                Vec3f lightDir, lightIntensity;
                IntersectInfo isectShad;
                lights[i]->illuminate(hitPoint, lightDir, lightIntensity, isectShad.tNear);
                bool visible = !trace(hitPoint + hitNormal * options.bias, -lightDir, objects, isectShad, kShadowRay);
                directLighting = visible * lightIntensity * max(0.0f, hitNormal.dotProduct(-lightDir));
            }

            Vec3f indirectLighting = Vec3f(0);

            uint32_t N = 128;//no of samples

            Vec3f Nt, Nb;

            createCoordinateSystem(hitNormal, Nt, Nb);

            float pdf = 1 / (2 * M_PI);

            for (uint32_t n = 0; n < N; n++) {

                float r1 = distribution(generator);
                float r2 = distribution(generator);

                Vec3f sample = uniformSampleHemisphere(r1, r2);

                Vec3f sampleWorld(sample.x * Nb.x + sample.y * hitNormal.x + sample.z * Nt.x,
                    sample.x * Nb.y + sample.y * hitNormal.y + sample.z * Nt.y,
                    sample.x * Nb.z + sample.y * hitNormal.z + sample.z * Nt.z);

                indirectLighting += r1 * castRay(hitPoint + sampleWorld * options.bias, sampleWorld, objects, lights, options, depth + 1) / pdf;
            }
            indirectLighting /= (float)N;

            hitColor = (directLighting / M_PI + 2 * indirectLighting) * isect.hitObject->albedo;
            break;
        }

        default:
            break;
        }
    }

    else {
        hitColor = 0;
    }

    return hitColor;
}

void render(const Options& options, const vector<unique_ptr<Object> >& objects, const vector<unique_ptr<Light> >& lights)
{

    unique_ptr<Vec3f[]> framebuffer(new Vec3f[options.width * options.height]);
    Vec3f* pixel = framebuffer.get();

    float scale = tan(degToRad(options.fieldOfView * 0.5f));

    float imageAspectRatio = (float)options.width / (float)options.height;

    Vec3f origin;

    options.cameraToWorld.multVecMatrix(Vec3f(0), origin);

    auto timeStart = std::chrono::high_resolution_clock::now();

    for (uint32_t j = 0; j < options.height; j++) {

        for (uint32_t i = 0; i < options.width; i++) {

            //generate primary ray direction

            float x = (2 * (i + 0.5f) / (float)options.width - 1) * imageAspectRatio * scale;
            float y = (1 - 2 * (j + 0.5f) / (float)options.height) * scale;

            Vec3f dir;
            options.cameraToWorld.multDirMatrix(Vec3f(x, y, -1), dir);
            dir.normalize();
            *(pixel++) = castRay(origin, dir, objects, lights, options);
        }

        fprintf(stderr, "\r%3d%c", uint32_t(j / (float)options.height * 100), '%');
    }
    auto timeEnd = std::chrono::high_resolution_clock::now();
    auto passedTime = std::chrono::duration<double, std::milli>(timeEnd - timeStart).count();
    fprintf(stderr, "\rDone: %.2f (sec)\n", passedTime / 1000);

    //save framebuffer to File

    float gamma = 1;

    ofstream ofs;
    ofs.open("out2.ppm");
    ofs << "P6\n" << options.width << " " << options.height << "\n255\n";
    for (uint32_t i = 0; i < options.height * options.width; i++) {

        char r = (char)(255 * clamp(0, 1, powf(framebuffer[i].x, 1.0f / gamma)));
        char g = (char)(255 * clamp(0, 1, powf(framebuffer[i].y, 1.0f / gamma)));
        char b = (char)(255 * clamp(0, 1, powf(framebuffer[i].z, 1.0f / gamma)));

        ofs << r << g << b;
    }
    ofs.close();
}

int main()
{

    vector<unique_ptr<Object> > objects;
    vector<unique_ptr<Light> > lights;
    Options options;

    options.fieldOfView = 39.89f;
    options.width = 512;
    options.height = 512;
    options.cameraToWorld = Matrix44f(0.965926f, 0.0f, -0.258819f, 0.0f, 0.0066019f, 0.999675f, 0.0246386f, 0.0f, 0.258735f, -0.0255078f, 0.965612f, 0.0f, 0.764985f, 0.791882f, 5.868275f, 1.0f);

    Matrix44f k1(2,0,0,0,
                0,2,0,0,
                0,0,2,0,
                0,0,0,1);

    TriangleMesh* plane = loadPolyMeshFromFile("./planegi.geo", k1);
    if (plane != nullptr) {
        plane->albedo = Vec3f(0.225f, 0.114f, 0.114f);
        objects.push_back(unique_ptr<Object>(plane));
    }
   Matrix44f kR(2,0,0,0,
                0,0,-1,0,
                0,1,0,0,
                0,1,-2,1);
   TriangleMesh* plane2 = loadPolyMeshFromFile("./planegi.geo", kR);
   if (plane != nullptr) {
       plane->albedo = Vec3f(0.225f, 0.114f, 0.114f);
       objects.push_back(unique_ptr<Object>(plane2));
   }

    TriangleMesh* cube = loadPolyMeshFromFile("./cubegi.geo", kIdentity);
    if (cube != nullptr) {
        cube->albedo = Vec3f(0.188559f, 0.287f, 0.200726f);
        objects.push_back(unique_ptr<Object>(cube));
    }

    Matrix44f xformSphere;
    xformSphere[3][1] = 1;

    Sphere* sph = new Sphere(xformSphere, 1);
    objects.push_back(unique_ptr<Object>(sph));

    Matrix44f l2w(0.916445f, -0.218118f, 0.335488f, 0.0f, 0.204618f, -0.465058f, -0.861309f, 0.0f, 0.343889f, 0.857989f, -0.381569f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    lights.push_back(unique_ptr<Light>(new DistantLight(l2w, 1, 16)));

    render(options, objects, lights);

    return 0;
}
