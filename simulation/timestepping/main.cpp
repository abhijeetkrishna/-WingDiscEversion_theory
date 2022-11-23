#define SE_CLASS1  // always use it
#define PRINT_STACKTRACE
#define STOP_ON_ERROR

#include "Vector/vector_dist.hpp"
#include "Operators/Vector/vector_dist_operators.hpp"
#include "OdeIntegrators/OdeIntegrators.hpp"
#include "hash_map/hopscotch_map.h"
#include "hash_map/hopscotch_set.h"
#include <sys/stat.h>
#include <string>


/*struct state_type_6d_ofp{
    state_type_5d_ofp(){
    }
    typedef size_t size_type;
    typedef int is_state_vector;
    aggregate<texp_v<double>,texp_v<double>,texp_v<double>,texp_v<double>,texp_v<double>,texp_v<double>> data;

    size_t size() const
    { return data.get<0>().size(); }

    void resize(size_t n)
    {
        data.get<0>().resize(n);
        data.get<1>().resize(n);
        data.get<2>().resize(n);
        data.get<3>().resize(n);
        data.get<4>().resize(n);
        data.get<5>().resize(n);
    }
};
namespace boost {
    namespace numeric {
        namespace odeint {
            template<>
            struct is_resizeable<state_type_6d_ofp> {
            typedef boost::true_type type;
            static const bool value = type::value;
            };
            }
        }
    }

template<>
struct vector_space_norm_inf<state_type_6d_ofp>
{
    typedef double result_type;
};*/


//main program

constexpr int x = 0;
constexpr int y = 1;
constexpr int z = 2;

constexpr int mass       =     0;
constexpr int velocity   =       1;
constexpr int force      =       2;
constexpr int pid        =    3;
constexpr int nbs        =    4;
constexpr int onbs       =    5;
constexpr int length     =    6;
constexpr int n_length   =    7;
constexpr int k_s       =    8;
constexpr int n_s       =    9;
constexpr int active       =    10;
constexpr int spring_active    =   11;
constexpr int frame     =   12;
constexpr int n_nbs     =   13;
constexpr int ext_force = 14;
constexpr int mass_density = 15;



constexpr int nbs_max  =       18;

int n_particles = 0;
double n_springs = 0;

double Dt;


struct ModelCustom
{
    template<typename Decomposition, typename vector> inline void addComputation(Decomposition & dec,
                                                                                 vector & particles,
                                                                                 size_t cell_id,
                                                                                 size_t p)
    {
            dec.addComputationCost(cell_id,300);
    }

    template<typename Decomposition> inline void applyModel(Decomposition & dec, size_t v)
    {
    }

    double distributionTol()
    {
        return 1.01;
    }
};

struct node_sim
{
    typedef boost::fusion::vector<float[3]> type;

    type data;

    struct attributes
    {
        static const std::string name[];
    };

    typedef float s_type;

    node_sim() {};

    static const unsigned int max_prop = 1;
};

const std::string node_sim::attributes::name[] = {"x"};

template<typename particle_type>
void write_connection_vtk(std::string file,particle_type & part, int n)
{
    std::cout << "writing vtk!" << std::endl;
    Graph_CSR<node_sim,aggregate<int>> graph;
    
    auto it = part.getDomainAndGhostIterator();
    
    while(it.isNext())
    {
        auto p = it.get();
        
        //std::cout << part.getPos(p)[x] << std::endl;
        //std::cout << pid << std::endl;

        graph.addVertex();
        graph.vertex(p.getKey()).template get<0>()[x] = part.getPos(p)[x];
        graph.vertex(p.getKey()).template get<0>()[y] = part.getPos(p)[y];
        graph.vertex(p.getKey()).template get<0>()[z] = part.getPos(p)[z];
        
        
        ++it;
        //std::cout << part.getPos(p)[x] << std::endl;
    }
    //std::cout << "3" << std::endl;

    auto it2 = part.getDomainIterator();

    while(it2.isNext())
    {
        auto p = it2.get();

        for (int j = 0 ; j < part.template getProp<n_nbs>(p) ; j++) {
            auto nnp = part.template getProp<onbs>(p)[j];

            graph.addEdge(p.getKey(),nnp);
        }

        ++it2;
    }

    VTKWriter<Graph_CSR<node_sim,aggregate<int>>,VTK_GRAPH> gv2(graph);
    gv2.write(file + "_" + std::to_string(create_vcluster().rank()) + "_" + std::to_string(n) + ".vtk");
}

template<typename particles_type>
void reconnect(particles_type & part)
{
    part.template ghost_get<pid>();

    auto it2 = part.getDomainAndGhostIterator();

    tsl::hopscotch_map<int,int> map;

    while(it2.isNext())
    {
        auto p = it2.get();

        map[part.template getProp<pid>(p)] = p.getKey();

        ++it2;
    }

    auto it = part.getDomainIterator();

    while(it.isNext())
    {
        auto p = it.get();

        for (int j = 0 ; j < part.template getProp<n_nbs>(p) ; j++)
        {
            int pid = part.template getProp<nbs>(p)[j];

            auto fnd = map.find(pid);
            if (fnd == map.end())
            {
                std::cout << "RECONNECTION FAILED " << pid << std::endl;
                part.write("crash");
                exit(1);
            }
            else
            {
                part.template getProp<onbs>(p)[j] = fnd->second;
            }
        }
        
    
        
        ++it;
    }
    
    std::cout << "reconnected!" << std::endl;
}

void *vectorGlobal;
std::string dir;

typedef vector_dist<3, double, aggregate<double, VectorS<3,double>, VectorS<3,double>, int, VectorS<nbs_max,int>, VectorS<nbs_max,int>, VectorS<nbs_max,double>, VectorS<nbs_max,double>, VectorS<nbs_max,double>, VectorS<nbs_max,double>, int, VectorS<nbs_max,int>, int, int, VectorS<3,double>, bool, double>> vector_type;


void RHS( const state_type_3d_ofp &stateV , state_type_3d_ofp &statedVdt , const double t )
{
    vector_type &particles= *(vector_type *) vectorGlobal;
    auto V=getV<velocity>(particles);
    auto F=getV<force>(particles);
    auto M=getV<mass>(particles);
    V[x]=stateV.data.get<x>();
    V[y]=stateV.data.get<y>();
    V[z]=stateV.data.get<z>();
  
    //only gives position, if we use any other properties
    //need to do ghost_get<prop1,prop2>
    particles.ghost_get<>(SKIP_LABELLING);
        
        //double strain = 0;
        auto it2= particles.getDomainIterator();
        while(it2.isNext()){
            
            auto p = it2.get();
            
            Point<3,double> xp = particles.getPos(p);
            //size_t count  = 0;
            double fx=0 + particles.template getProp<ext_force>(p)[x];
            double fy=0 + particles.template getProp<ext_force>(p)[y];
            double fz=0 + particles.template getProp<ext_force>(p)[z];
            double x1=0;
            double y1=0;
            double z1=0;
            double xm=0;
            
            //this is tru for all particles that were on the barrier, corrects their position
            
            for (int j = 0 ; j < particles.getProp<n_nbs>(p) ; j++) {
                
                auto nnp = particles.getProp<onbs>(p)[j];
                Point<3, double> xq = particles.getPos(nnp);
                if (xp != xq) {
                    
                    x1=xq[x]-xp[x];
                    y1=xq[y]-xp[y];
                    z1=xq[z]-xp[z];
                    
                    xm=sqrt(x1*x1+y1*y1+z1*z1);
                    particles.template getProp<length>(p)[j] = xm;
                
                    //std::cout << particles.template getProp<pid>(p) << " " << particles.template getProp<pid>(nnp) << ": ";
                    //std::cout << particles.template getProp<length>(p)[j] << " " << particles.template getProp<n_length>(p)[j] << "\n";
                    
                    fx=fx+particles.template getProp<k_s>(p)[j]*(particles.template getProp<length>(p)[j]-particles.template getProp<n_length>(p)[j])*x1/xm-particles.template getProp<n_s>(p)[j]*particles.template getProp<velocity>(p)[x];
                    fy=fy+particles.template getProp<k_s>(p)[j]*(particles.template getProp<length>(p)[j]-particles.template getProp<n_length>(p)[j])*y1/xm-particles.template getProp<n_s>(p)[j]*particles.template getProp<velocity>(p)[y];
                    fz=fz+particles.template getProp<k_s>(p)[j]*(particles.template getProp<length>(p)[j]-particles.template getProp<n_length>(p)[j])*z1/xm-particles.template getProp<n_s>(p)[j]*particles.template getProp<velocity>(p)[z];
                    //count = count + 1;
                }
            }
            
            if(particles.template getProp<frame>(p) == 1){
                particles.template getProp<force>(p)[x]=0;
                particles.template getProp<force>(p)[y]=0;
                particles.template getProp<force>(p)[z]=0;
             }
            else {
                particles.template getProp<force>(p)[x]=fx;///particles.template getProp<mass>(p);
                particles.template getProp<force>(p)[y]=fy;///particles.template getProp<mass>(p);
                particles.template getProp<force>(p)[z]=fz;///particles.template getProp<mass>(p);
            }
            //std::cout << "t= " << t << "; " << particles.template getProp<pid>(p) << "; f= ";
            //std::cout << fx << " " << fy << " " << fz << "\n";

            ++it2;
        }
        statedVdt.data.get<x>()=F[x]/M;
        statedVdt.data.get<y>()=F[y]/M;
        statedVdt.data.get<z>()=F[z]/M;
}

int ctr=1;
double t_old;
double t_csv = 0;
double t_vtk = 0;
double save_csv, save_vtk;
void MoveAndWatch(state_type_3d_ofp &stateV , const double t )
{
    vector_type &particles= *(vector_type *) vectorGlobal;
    auto V=getV<velocity>(particles);
    auto Pos = getV<PROP_POS>(particles);
    auto &v_cl=create_vcluster<>();
    V[x]=stateV.data.get<x>();
    V[y]=stateV.data.get<y>();
    V[z]=stateV.data.get<z>();
    
    if (t!=0){
        Pos=Pos+(t-t_old)*V;
        
        if (ctr % 1000 == 0)
        {
            ModelCustom md;
            particles.addComputationCosts(md);
            particles.getDecomposition().decompose();
            particles.map();
            particles.ghost_get<>();
            reconnect(particles);
        }
    //Add things here that are required after moving
        auto it3 = particles.getDomainIterator();
        while(it3.isNext()){
            auto p = it3.get();
            //add hard surface below
            if( particles.getPos(p)[z] < 0){
            particles.getPos(p)[z] = -particles.getPos(p)[z];
            }
            ++it3;
        }

    }
    if (v_cl.rank() == 0) {
                std::cout << "Time step " << ctr << " : " << t << " over." << "dt =" <<t-t_old<< std::endl;
                std::cout << "----------------------------------------------------------" << std::endl;
        }
    
    
    //write vtk and csv frames
    t_csv = t_csv + (t-t_old);
    t_vtk = t_vtk + (t-t_old);
    
    if (t_vtk > save_vtk) {
        int time_int = t;
        write_connection_vtk(dir + "/sim_output/connections",particles,time_int);
        t_vtk = 0;
    }
    
    if (t_csv > save_csv) {
        int time_int = t;
        particles.write_frame(dir + "/sim_output/Spring",time_int,CSV_WRITER);
        t_csv = 0;
    }
    
    ctr++;
    t_old=t;
    

    stateV.data.get<x>()=V[x];
    stateV.data.get<y>()=V[y];
    stateV.data.get<z>()=V[z];
    

}

int main(int argc, char * argv[])
{
    dir = argv[1];
    
    
    //read simulation parameters from .csv
    std::ifstream param_file(dir + "/runfiles/sim_params.csv");
    std::string param_line;
    
    
    std::getline(param_file, param_line);
    std::stringstream ss(param_line);
    
    double dt, tf, dim;

    //int nbs_m;
    
    ss >> dt;
    ss >> tf;
    ss >> save_csv;
    ss >> save_vtk;
    ss >> dim;
    
    
    //int save_csv = save_csv_d/dt;
    //int save_vtk = save_vtk_d/dt;
    
    std::cout << "dt " << dt << "\n";
    std::cout << "tf " << tf << "\n";
    std::cout << "save_csv " << save_csv << "\n";
    std::cout << "save_vtk " << save_vtk << "\n";
    std::cout << "dim " << dim << "\n";
    
    size_t psz=20;
    const size_t sz[3] = {psz,psz,2};
        Box<3, double> box({-dim, -dim, -dim}, {dim, dim, dim});
        size_t bc[3] = {NON_PERIODIC,NON_PERIODIC,NON_PERIODIC};
        double spacing = 1.0;
        Ghost<3, double> ghost(spacing * 3);

    grid_sm<3,void> g(sz);
    
    size_t cell_sz[] = {32,32,1};
    grid_sm<3,void> g_cell(cell_sz);
    
    openfpm_init(&argc,&argv);
    //                                 mass    velocity    force       pid        nbs         onbs          length           n_length             k_s             n_s              active spring_active frame n_nbs ext_force
    
    vector_type particles(0, box, bc, ghost,g_cell);

    
    auto & v_cl = create_vcluster<>();
    particles.setPropNames({"mass","velocity","force","pid","nbs","onbs","length","n_length","k_s","n_s","active","spring_active","frame","n_nbs", "ext_force", "mass_density"});
    
    if (v_cl.rank() == 0)
    {
        // Create an input filestream
        std::ifstream ball_file(dir + "/runfiles/balls.csv");
        std::ifstream neighbours_file(dir + "/runfiles/neighbours.csv");
        std::ifstream l0_file(dir + "/runfiles/neigh_l0.csv");
        std::ifstream k_file(dir + "/runfiles/neigh_k.csv");
        std::ifstream ns_file(dir + "/runfiles/neigh_viscoelastic_coeff.csv");

        std::string line;
        std::string line_neigh;
        std::string line_l0;
        std::string line_k;
        std::string line_ns;
        
        while(std::getline(ball_file, line)) {
            //std::cout << line << "\n";
            std::getline(neighbours_file, line_neigh);
            std::getline(l0_file, line_l0);
            std::getline(k_file, line_k);
            std::getline(ns_file, line_ns);

            std::stringstream ss(line);
            std::stringstream ss_neigh(line_neigh);
            std::stringstream ss_l0(line_l0);
            std::stringstream ss_k(line_k);
            std::stringstream ss_ns(line_ns);

            particles.add();

            double g_id;
            ss >> g_id;

            particles.getLastProp<pid>() = g_id;

            for(int i=0; i<3; i++) {
                double coord;
                ss >> coord;
                particles.getLastPos()[i] = coord;
            }

            double n_neigh;
            ss >> n_neigh;
            particles.getLastProp<n_nbs>() = n_neigh;
            if(nbs_max < n_neigh){
                std::cout << "INCREASE NBS_MAX to " << n_neigh << "\n";
            }
            
            double part_mass;
            ss >> part_mass;
            //std::cout << line << "\n";
            particles.getLastProp<mass>() = part_mass;
            
            double part_mass_density;
            ss >> part_mass_density;
            std::cout << "mass_density " << part_mass_density << "\n";
            //std::cout << line << "\n";
            particles.getLastProp<mass_density>() = part_mass_density;

            double act;
            ss >> act;
            particles.getLastProp<active>() = act;
            
            double frame_bool;
            ss >> frame_bool;
            particles.getLastProp<frame>() = frame_bool;
            
            for(int i=0; i<3; i++) {
                double ext_force_input;
                ss >> ext_force_input;
                particles.getLastProp<ext_force>()[i] = ext_force_input;
            }

            //double frame_nb1, frame_nb2;
            //ss >> frame_nb1;
            //ss >> frame_nb2;
            
            //particles.getLastProp<frame_nbs>()[0] = int(frame_nb1);
            //particles.getLastProp<frame_nbs>()[1] = int(frame_nb2);
            
            n_springs = n_springs + n_neigh;

            for(int i = 0; i < n_neigh; i++){
                
                double neigh, l0, k, ns;
                int act;
                ss_neigh >> neigh;
                ss_l0 >> l0;
                ss_k >> k;
                ss_ns >> ns;


                particles.getLastProp<nbs>()[i] = neigh;
                particles.getLastProp<n_length>()[i] = l0;
                particles.getLastProp<k_s>()[i] = k;
                particles.getLastProp<n_s>()[i] = ns;

            }

            particles.getLastProp<velocity>()[x]=0;
            particles.getLastProp<velocity>()[y]=0;
            particles.getLastProp<velocity>()[z]=0;
            

            ++n_particles;
        }
    }

    particles.map();
    particles.ghost_get<pid>();
    
    reconnect(particles);
    //
    
    //particles.write_frame(dir + "test",0,CSV_WRITER);
    //std::cout << "min_mass " << min_mass << "\n";

    ModelCustom md;
    particles.addComputationCosts(md);
    particles.getDecomposition().decompose();
    particles.map();
    
    
    reconnect(particles);
    
    timer timer;
    timer.start();
    vectorGlobal = (void *) &particles;


    auto V=getV<velocity>(particles);
    V=0;
    //openfpm_finalize();
    //return 0;
    

    double t=0;
    //particles.write_frame("Spring",0,CSV_WRITER);
        
    //Odeint
     state_type_3d_ofp tV;

    tV.data.get<0>()=V[x];
    tV.data.get<1>()=V[y];
    tV.data.get<2>()=V[z];
    

    write_connection_vtk(dir + "/sim_output/connections",particles,0);
    particles.write_frame(dir + "/sim_output/Spring",0,CSV_WRITER);

    double tim=0;
    
    boost::numeric::odeint::runge_kutta4<state_type_3d_ofp, double, state_type_3d_ofp, double, boost::numeric::odeint::vector_space_algebra_ofp> Odeint_rk4;
    
    size_t steps = boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled( 1.0e-9, 1.0e-9 ,boost::numeric::odeint::runge_kutta_cash_karp54< state_type_3d_ofp,double,state_type_3d_ofp,double,boost::numeric::odeint::vector_space_algebra_ofp>() ) , RHS , tV , tim , tf , dt, MoveAndWatch);
    
    //size_t steps = boost::numeric::odeint::integrate_const(Odeint_rk4, RHS, tV, tim, tf, dt, MoveAndWatch);
    V[x]=tV.data.get<0>();
    V[y]=tV.data.get<1>();
    V[z]=tV.data.get<2>();
    
    
    particles.ghost_get<force>(SKIP_LABELLING);
    particles.write_frame(dir + "/sim_output/Spring_final",tf,CSV_WRITER);
    write_connection_vtk(dir + "/sim_output/connections_final",particles,tf);
    
    std::ofstream outf(dir + "/sim_output/sim_params.txt");
    
    outf << "Simulation completed with the following parameters:" << "\n";
    outf << "number of balls: " << n_particles << "\n";
    outf << "number of springs: " << n_springs << "\n";
    outf << "time: " << tf << "\n";
    outf << "dt: " << dt << "\n";
    //outf << "mass: " << part_mass << "\n";
    outf << "box dimensions: " << dim << "\n";
    outf << "running time: " << timer.getwct() << "\n";
    outf << "steps: " << ctr << "\n";
    
    openfpm_finalize();
    return 0;
}  

