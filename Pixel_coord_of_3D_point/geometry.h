#include <iostream.h>
#include <iomanip.h>
#include<stdio.h>
#include <stdlib.h>
#include <math.h>

template<typename T>
class Vec2{
		public:
			Vec2() : x(T(0)),y(T(0)){}
			Vec2(T xx) : x(xx),y(xx){}
			Vec2(T xx,T yy): x(xx),y(yy){}

			Vec2 operator + (const Vec2 &v)const
			{return Vec2(x+v.x,y+v.y);}

			Vec2 operator - (const Vec2 &v)const
			{return Vec2(x-v.x,y-yv.y);}
			Vec2 operator - () const
			{return Vec2(-x,-y,-z);}
			Vec2 operator * (const T &r){
				return Vec2(x*r,y*r);
			}
			Vec2 operator * (const Vec2 &v){
				return Vec2(x*v*x,y*v.y);
			}
			Vec2 operator *= (const T &r){
				x*=r;y*=r;
				return *this;
			}

			Vec2 operator /= (const T &r){
				x/=r;y/=r;
				return *this;
			}
			T dotProduct(const Vec2<T> &v){
				return(x*v.x+y*v.y);
			}
			friend std::ostream& operator << (std::ostream &s,const Vec2 &v){
				return s << '[' << v.x << ' ' << v.y << ']';
			}
			friend Vec2 operator * (const T &r , const Vec2 &v){
				return Vec2(v.x*r,v.y*r);
			}
			T x,y;

};

typedef Vec2<float> Vec2f;
typedef Vec2<int> Vec2i;

template<typename T>
class Vec3{
		public:
			Vec3() : x(T(0)),y(T(0)),z(T(0)){}
			Vec3(T xx) : x(xx),y(xx),z(xx){}
			Vec3(T xx,T yy,T zz): x(xx),y(yy),z(zz){}

			Vec3 operator + (const Vec3 &v) const
			{return Vec3(x+v.x,y+v.y,z+v.z);}
			Vec3 operator - (const Vec3 &v) const
			{return Vec3(x-v.x,y-v.y,z-v.z);}
			Vec3 operator - () const
			{return Vec3(-x,-y,-z);}
			Vec3 operator * (const T &r) const
			{return Vec3(x*r,y*r,z*r);}
			Vec3 operator * (const Vec3 &v) const
			{return Vec3(x*v.x,y*v.y,z*v.z);}
			T dotProduct(const Vec3<T> &v){
				return(x*v.x+y*v.y+z*v.z);
			}
			Vec3& operator *= (const T &r){
				x*=r;y*=r;z*=r;return *this;
			}
			Vec3& operator /= (const T &r){
				x/=r;y/=r;z/=r;return *this;
			}
			Vec3 crossProduct(const Vec3 &v) const{
				return Vec3((y*v.z-z*v.y),(x*v.z-z*v.x)*(-1),(x*v.y-y*v.x));
			}
			T norm() const
			{return(x*x+y*y+z*z);}
			T length() const
			{return sqrt(norm());}

			const T& operator [](unint8_t i){return (&x)[i];}
			T& operator [](unint8_t i){return (&x)[i];}

			Vec3& normalize(){
				T n = norm();
				if(n>0){
					T factor = (1/sqrt(n));
					x*=factor;y*=factor;z*=factor;
				}
				return *this;
			}

			friend Vec3 operator * (const T &r, const Vec3 &v){
				return Vec3<T>(v.x*r,v.y*r,v.z*r);
			}

			friend Vec3 operator / (const T &r, const Vec3 &v){
				return Vec3<T>(v.x/r,v.y/r,v.z/r);
			}
			friend std::ostream& operator << (std::ostream &s, const Vec3<T> &v){
				return s << '[' << v.x << ' ' << v.y << ' ' << v.z <<']';
			}
			 T x,y,z;

};

typedef Vec3<float> Vec3f;
typedef Vec3<int> Vec3i;
