#function render(wall::Wall, ax; kwargs...)
    #plot([wall.a.x, wall.b.x], [wall.a.y, wall.b.y]; kwargs...)
#end
#
#function render(scene::Scene, ax; kwargs...)
    #for wall in scene.walls
        #render(wall, ax; kwargs...)
    #end
    #xspan = scene.bounds.xmax - scene.bounds.xmin
    #yspan = scene.bounds.ymax - scene.bounds.ymin
    #ax[:set_xlim]((scene.bounds.xmin - xspan * 0.05, scene.bounds.xmax + xspan * 0.05))
    #ax[:set_ylim]((scene.bounds.ymin - yspan * 0.05, scene.bounds.ymax + yspan * 0.05))
#end
function render_tree(tree::RRTTree, thickness)
    for node in 1:tree.num_nodes
        a = Point(tree.confs[1,node], tree.confs[2,node])
        parent = tree.parents[node]
        if parent == 0
            continue
        end
        b = Point(tree.confs[1,parent],tree.confs[2,parent])
        plot([a.x, b.x], [a.y, b.y], color="black", linewidth=thickness)
    end
end
